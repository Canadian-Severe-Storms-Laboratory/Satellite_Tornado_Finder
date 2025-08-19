using ArcGIS.Core.CIM;
using ArcGIS.Core.Geometry;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Mapping;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using static ArcGISUtils.Utils;
using System.Threading.Tasks.Dataflow;
using System.Collections.Concurrent;
using System.Linq;
using Newtonsoft.Json;
using BitMiracle.LibTiff.Classic;


namespace Satellite_Analyzer
{
    public struct SearchResult(int x, int y, int s, int c)
    {
        public int tileX = x, tileY = y, score = s, count = c;

        public override readonly string ToString()
        {
            return string.Format("{0}, {1} - {2} ({3})", tileX, tileY, score, count);
        }
    }

    public class SystematicSearch
    {
        private static string CreateResultsFolder()
        {
            string path = GetProjectPath() + "\\SatelliteAnalysis";
            if (!Directory.Exists(path)) Directory.CreateDirectory(path);

            string dateTime = DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss");

            path += "\\Results_" + dateTime;

            if (!Directory.Exists(path)) Directory.CreateDirectory(path);

            return path;
        }

        public static string GetSavePath()
        {
            return savePath;
        }

        private static PlanetReader planetReader;
        private static TornadoPatchPredictor tpp;
        private static TornadoPatchPredictor64 tpp64;
        private static List<(int, int)> tiles;
        private static List<SearchResult> foundTiles;
        private static string savePath;
        private static GroupLayer predGroup;
        //private static GroupLayer diffGroup;
        private static Monitor monitor;

        private static async Task InitalizeSearch(Polygon polygon)
        {
            OpenConsole();

            planetReader = new();
            await planetReader.BuildBaseMapDict();

            await LandCover.Initalize();

            tiles = PolygonToTiles(polygon);

            foundTiles = [];

            tpp = new(AddinAssemblyLocation() + "\\tornado_patch_predictor_de_norm.onnx");
            tpp64 = new(AddinAssemblyLocation() + "\\model64.onnx");

            savePath = CreateResultsFolder();

            predGroup = await QueuedTask.Run(() => { return LayerFactory.Instance.CreateGroupLayer(MapView.Active.Map, 0, "Tornado_Prediction"); });
            //diffGroup = await QueuedTask.Run(() => { return LayerFactory.Instance.CreateGroupLayer(MapView.Active.Map, 0, "Differnce_Images"); });

            monitor = new("Search Progress", tiles.Count);
        }

        public static async Task<List<SearchResult>> Search(Polygon polygon, int bMonth, int bYear, int aMonth, int aYear)
        {
            CIMColorRamp aspectRamp = await QueuedTask.Run(() =>
            {
                StyleProjectItem style = ArcGIS.Desktop.Core.Project.Current.GetItems<StyleProjectItem>().FirstOrDefault(s => s.Name.Equals("ArcGIS Colors", StringComparison.OrdinalIgnoreCase));
                ColorRampStyleItem aspectItem = style?.SearchColorRamps("Aspect").FirstOrDefault();
                return aspectItem.ColorRamp;
            });

            await InitalizeSearch(polygon);

            ConcurrentBag<SearchResult> significantTiles = [];

            monitor.Start();

            var downloadBlock = new TransformBlock<(int, int), (int, int, byte[], byte[], int, Envelope, byte[], byte[], int)>(async (tile) =>
            {
                try
                {
                    if (monitor.cancelled) return (tile.Item1, tile.Item2, null, null, 0, null, null, null, 0);

                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    var (x, y) = tile;

                    string imgName = $"_{x}_{y}.png";
                    string beforePath = savePath + "\\before" + imgName;
                    string afterPath = savePath + "\\after" + imgName;

                    var (beforeBytes, beforeUDMBytes, envelope, beforeMaskType) = await planetReader.FindImage(x, y, bMonth, bYear, beforePath);
                    if (beforeBytes == null) return (x, y, null, null, 0, null, null, null, 0);

                    var (afterBytes, afterUDMBytes, _, afterMaskType) = await planetReader.FindImage(x, y, aMonth, aYear, afterPath);
                    if (afterBytes == null) return (x, y, null, null, 0, null, null, null, 0);

                    watch.Stop();
                    Console.WriteLine($"Tile {x}, {y} downloaded in {watch.ElapsedMilliseconds} ms");

                    return (x, y, beforeBytes, beforeUDMBytes, beforeMaskType, envelope, afterBytes, afterUDMBytes, afterMaskType);
                }
                catch (Exception ex)
                {
                    System.Windows.MessageBox.Show("Error in Download Block:\n\n" + ex.Message);
                }

                return (tile.Item1, tile.Item2, null, null, 0, null, null, null, 0);
            },
            new ExecutionDataflowBlockOptions
            {
                BoundedCapacity = 4,
                EnsureOrdered = false
            });

            var preprocessBlock = new TransformBlock<(int, int, byte[], byte[], int, Envelope, byte[], byte[], int), (int, int, Mat, Mat, byte[], Envelope, Mat, Mat, byte[])>((packet) =>
            {
                try
                {
                    var watch = System.Diagnostics.Stopwatch.StartNew();

                    var (x, y, beforeBytes, beforeUDMBytes, beforeMaskType, envelope, afterBytes, afterUDMBytes, afterMaskType) = packet;

                    if (monitor.cancelled || beforeBytes == null || afterBytes == null) return (x, y, null, null, null, null, null, null, null);

                    Mat beforeImg = Cv2.ImDecode(beforeBytes, ImreadModes.Color);
                    Mat beforeCCImg = PlanetReader.DecodeUDM(beforeUDMBytes, beforeMaskType);
                    Mat afterImg = Cv2.ImDecode(afterBytes, ImreadModes.Color);
                    Mat afterCCImg = PlanetReader.DecodeUDM(afterUDMBytes, afterMaskType);

                    Cv2.MedianBlur(beforeImg, beforeImg, 5);
                    Cv2.MedianBlur(afterImg, afterImg, 5);

                    watch.Stop();
                    Console.WriteLine($"Tile {x}, {y} preprocessed in {watch.ElapsedMilliseconds} ms");

                    return (x, y, beforeImg, beforeCCImg, beforeBytes, envelope, afterImg, afterCCImg, afterBytes);
                }
                catch (Exception ex)
                {
                    System.Windows.MessageBox.Show("Error in Pre Processing Block:\n\n" + ex.Message);
                }

                return (packet.Item1, packet.Item2, null, null, null, null, null, null, null);
            },
            new ExecutionDataflowBlockOptions
            {
                BoundedCapacity = 4,
                EnsureOrdered = false
            });

            var predictionBlock = new TransformBlock<(int, int, Mat, Mat, byte[], Envelope, Mat, Mat, byte[]), (int, int, Mat, Mat, byte[], Envelope, Mat, Mat, byte[], ByteVector, ByteVector)>((packet) =>
            {
                try
                {
                    var watch = System.Diagnostics.Stopwatch.StartNew();

                    var (x, y, beforeImg, beforeCCImg, beforeBytes, envelope, afterImg, afterCCImg, afterBytes) = packet;

                    if (monitor.cancelled || beforeImg == null || afterImg == null) return (x, y, null, null, null, null, null, null, null, null, null);

                    ByteVector tornadoPredictionA = tpp.analyze(beforeImg, afterImg, beforeImg.Width, beforeImg.Height);
                    ByteVector tornadoPredictionB = tpp64.analyze(beforeImg, afterImg, beforeImg.Width, beforeImg.Height);

                    watch.Stop();
                    Console.WriteLine($"Tile {x}, {y} predicted in {watch.ElapsedMilliseconds} ms");

                    return (x, y, beforeImg, beforeCCImg, beforeBytes, envelope, afterImg, afterCCImg, afterBytes, tornadoPredictionA, tornadoPredictionB);
                }
                catch (Exception ex)
                {
                    System.Windows.MessageBox.Show("Error in Prediction Block:\n\n" + ex.Message);
                }

                return (packet.Item1, packet.Item2, null, null, null, null, null, null, null, null, null);
            }, 
            new ExecutionDataflowBlockOptions
            {
                BoundedCapacity = 1,
                EnsureOrdered = false
            });

            var postProcessBlock = new TransformBlock<(int, int, Mat, Mat, byte[], Envelope, Mat, Mat, byte[], ByteVector, ByteVector), (int, int, int, int, Mat, Mat, Mat, Mat, byte[])> (async (packet) =>
            {
                try 
                { 
                    if (monitor.cancelled) return (packet.Item1, packet.Item2, 0, 0, null, null, null, null, null);
                    var watch = System.Diagnostics.Stopwatch.StartNew();

                    var (x, y, beforeImg, beforeCCImg, beforeBytes, envelope, afterImg, afterCCImg, afterBytes, tornadoPredictionA, tornadoPredictionB) = packet;

                    if (beforeImg == null || afterImg == null) return (x, y, 0, 0, null, null, null, null, null);

                    Mat predAImg = ByteVector.ToMat(tornadoPredictionA, beforeImg.Size());
                    Mat predBImg = ByteVector.ToMat(tornadoPredictionB, beforeImg.Size());

                    Mat predImg = FloatMulNormalized(predAImg, predBImg);

                    Mat landcoverImg = await LandCoverMask(envelope, beforeImg.Size());

                    Mat colorChangeMask = ColorChangeMask(beforeImg, afterImg);

                    Mat mask = new();
                    Cv2.BitwiseAnd(colorChangeMask, landcoverImg, mask);
                    Cv2.BitwiseAnd(mask, beforeCCImg, mask);
                    Cv2.BitwiseAnd(mask, afterCCImg, mask);

                    Mat diffImg = AbsDifferenceImage(beforeImg, afterImg);
                    Cv2.BitwiseAnd(mask, diffImg, diffImg);

                    Cv2.CvtColor(mask, mask, ColorConversionCodes.BGR2GRAY);

                    Cv2.BitwiseAnd(predImg, mask, predImg);
                    //Cv2.MorphologyEx(predImg, predImg, MorphTypes.Open, Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(15, 15)));
                    Cv2.MinMaxLoc(predImg, out double _, out double maxVal);

                    int pxCount = Cv2.CountNonZero(predImg.Threshold(Math.Max(maxVal-1, 0), 255, ThresholdTypes.Binary));

                    predImg = predImg.Threshold(127, 255, ThresholdTypes.Binary);

                    watch.Stop();
                    Console.WriteLine($"Tile {x}, {y} post processed in {watch.ElapsedMilliseconds} ms");

                    return (x, y, (int)maxVal, pxCount, predImg, diffImg, beforeImg, afterImg, afterBytes);
                }
                catch (Exception ex) {
                    System.Windows.MessageBox.Show( "Error in Post Processing Block:\n\n" + ex.Message );
                }

                return (packet.Item1, packet.Item2, 0, 0, null, null, null, null, null);
            },
            new ExecutionDataflowBlockOptions
            {
                BoundedCapacity = 4,
                EnsureOrdered = false
            });

            var saveBlock = new ActionBlock<(int, int, int, int, Mat, Mat, Mat, Mat, byte[])>(async (packet) =>
            {
                try
                {
                    if (monitor.cancelled) return;

                    var (x, y, maxVal, pxCount, predImg, diffImg, beforeImg, afterImg, afterBytes) = packet;

                    string imgName = $"_{x}_{y}.png";

                    if (maxVal < 128)
                    {
                        Console.WriteLine($"Tile {x}, {y} skipped");
                        monitor.Update();
                        return;
                    }

                    var watch = System.Diagnostics.Stopwatch.StartNew();

                    Cv2.ImWrite(savePath + "\\before" + imgName, beforeImg);
                    Cv2.ImWrite(savePath + "\\after" + imgName, afterImg);
                    Cv2.ImWrite(savePath + "\\diff" + imgName, diffImg);

                    Tiff pred = GeoTiff.CreateFromReference(afterBytes, savePath + $"\\pred_{x}_{y}.tif");
                    GeoTiff.WriteImage(pred, predImg);

                    significantTiles.Add(new(x, y, maxVal, pxCount));

                    await QueuedTask.Run(() =>
                    {
                        var layer = LoadRasterLayer(savePath, $"pred_{x}_{y}.tif", predGroup);

                        var c = layer.GetColorizer();

                        CIMRasterStretchColorizer colorizor = (CIMRasterStretchColorizer)layer.GetColorizer();
                        colorizor.StretchType = RasterStretchType.MinimumMaximum;
                        colorizor.DisplayBackgroundValue = true;
                        colorizor.ColorRamp = aspectRamp;
                        layer.SetColorizer(colorizor);
                    });

                    monitor.Update();
                    watch.Stop();
                    Console.WriteLine($"Tile {x}, {y} saved in {watch.ElapsedMilliseconds} ms");
                }
                catch (Exception ex) {
                    System.Windows.MessageBox.Show( "Error in Save Block:\n\n" + ex.Message );
                }
            },
            new ExecutionDataflowBlockOptions
            {
                BoundedCapacity = 4,
                EnsureOrdered = false
            });

            downloadBlock.LinkTo(preprocessBlock, new DataflowLinkOptions() { PropagateCompletion = true });
            preprocessBlock.LinkTo(predictionBlock, new DataflowLinkOptions() { PropagateCompletion = true });
            predictionBlock.LinkTo(postProcessBlock, new DataflowLinkOptions() { PropagateCompletion = true });
            postProcessBlock.LinkTo(saveBlock, new DataflowLinkOptions() { PropagateCompletion = true });

            try
            {
                foreach (var tile in tiles) await downloadBlock.SendAsync(tile);

                downloadBlock.Complete();

                await saveBlock.Completion;

                monitor.Stop();

                foundTiles = [.. significantTiles.OrderByDescending(x => x.score).ThenByDescending(x => x.count)]; //[.. significantTiles.OrderBy(x => x.tileY).ThenBy(x => x.tileX)];

                SaveSearchResults();
            }
            catch (Exception ex) {
                System.Windows.MessageBox.Show("Error in Search:\n\n" + ex.Message);
            }

            Console.WriteLine($"found {significantTiles.Count} of {tiles.Count}");

            return [.. foundTiles];
        }

        private static void SaveSearchResults()
        {
            string path = savePath + "\\results.json";

            Dictionary<string, object> results = new()
            {
                { "folderPath", savePath },
                { "tiles", foundTiles }
            };

            using StreamWriter sw = new(path);
            sw.WriteLine(JsonConvert.SerializeObject(results));

        }

        public static async Task<Mat> LandCoverMask(Envelope envelope, Size size)
        {
            Mat landcoverImg = LandCover.TypeMask(await LandCover.GetSection(envelope), LandCover.potentialForestTypes);

            Cv2.Resize(landcoverImg, landcoverImg, size, interpolation: InterpolationFlags.Cubic);
            Cv2.Threshold(landcoverImg, landcoverImg, 127, 255, ThresholdTypes.Binary);
            Cv2.Erode(landcoverImg, landcoverImg, Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(7, 7)), iterations: 2);
            Cv2.CvtColor(landcoverImg, landcoverImg, ColorConversionCodes.GRAY2BGR);

            return landcoverImg;
        }

        public static Mat ColorChangeMask(Mat before, Mat after, int threshold = 15)
        {
            Mat beforeRed = before.ExtractChannel(2);
            Mat afterRed = after.ExtractChannel(2);
            
            Mat diff = new();
            Cv2.Subtract(afterRed, beforeRed, diff);
            
            Mat mask = new();
            Cv2.Threshold(diff, mask, threshold, 255, ThresholdTypes.Binary);
            
            return mask.CvtColor(ColorConversionCodes.GRAY2BGR);
        }

        public static Mat AbsDifferenceImage(Mat before, Mat after)
        {
            Mat diffImg = new();

            Cv2.Absdiff(after, before, diffImg);

            //adjust brightness and contrast
            diffImg.ConvertTo(diffImg, -1, 30, -300);

            return diffImg;
        }

        public static Mat FloatMulNormalized(Mat imgA, Mat imgB)
        {
            Mat imgAFloat = new(); Mat imgBFloat = new(); Mat resultFloat = new();

            imgA.ConvertTo(imgAFloat, MatType.CV_32F, 1.0 / 255.0); 
            imgB.ConvertTo(imgBFloat, MatType.CV_32F, 1.0 / 255.0);

            Cv2.Multiply(imgAFloat, imgBFloat, resultFloat);

            Mat result = new(); 
            resultFloat.ConvertTo(result, MatType.CV_8U, 255.0);

            return result;
        }

        private static List<(int, int)> PolygonToTiles(Polygon polygon)
        {
            List<(int, int)> tiles = [];

            //Envelope polygonEnvelope = (Envelope)GeometryEngine.Instance.Project(polygon, SpatialReferences.WebMercator);

            Envelope polygonEnvelope = polygon.Extent;

            var (xmin, ymin) = PlanetReader.TileIndex(polygonEnvelope.YMin, -polygonEnvelope.XMin, mercator: true);
            var (xmax, ymax) = PlanetReader.TileIndex(polygonEnvelope.YMax, -polygonEnvelope.XMax, mercator: true);

            for (int y = (int)ymin; y <= (int)ymax + 1; y++)
            {
                for (int x = (int)xmin; x <= (int)xmax + 1; x++)
                {
                    Envelope tileEnvelope = PlanetReader.TileIndexToEnvelope(x, y);

                    if (GeometryEngine.Instance.Intersects(tileEnvelope, polygon)) tiles.Add((x, y));
                }
            }

            return tiles;
        }

        

    }
}

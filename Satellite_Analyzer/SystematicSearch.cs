using ArcGIS.Core.CIM;
using ArcGIS.Core.Geometry;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Mapping;
using BitMiracle.LibTiff.Classic;
using Newtonsoft.Json;
using OpenCvSharp;
using ScottPlot.TickGenerators.TimeUnits;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using static ArcGISUtils.Utils;


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
        private static TornadoPatchPredictor64_256 tpp64_256;
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
            tpp64_256 = new(AddinAssemblyLocation() + "\\model64_256.onnx");

            savePath = CreateResultsFolder();

            predGroup = await QueuedTask.Run(() => { return LayerFactory.Instance.CreateGroupLayer(MapView.Active.Map, 0, "Tornado_Prediction"); });
            //diffGroup = await QueuedTask.Run(() => { return LayerFactory.Instance.CreateGroupLayer(MapView.Active.Map, 0, "Differnce_Images"); });

            monitor = new("Search Progress", tiles.Count);
        }

        private static Mat MergeGrid3x3(Mat[] imgs)
        {
            int tileW = imgs[0].Width;
            int tileH = imgs[0].Height;
            Mat merged = new(tileH * 3, tileW * 3, imgs[0].Type());

            for (int row = 0; row < 3; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    int idx = row * 3 + col;
                    Rect roi = new(col * tileW, row * tileH, tileW, tileH);
                    imgs[idx].CopyTo(merged[roi]);
                }
            }

            // Crop a 1.25x region centered on the middle tile
            int cropW = (int)(tileW * 1.25);
            int cropH = (int)(tileH * 1.25);
            int centerX = tileW + tileW / 2;
            int centerY = tileH + tileH / 2;
            Rect cropRoi = new(centerX - cropW / 2, centerY - cropH / 2, cropW, cropH);

            Mat cropped = new(merged, cropRoi);
            Mat result = new();
            cropped.CopyTo(result);

            return result;
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

            var downloadBlock = new TransformBlock<(int, int), (int, int, Mat[], Mat[], Envelope, Mat[], Mat[])>(async (tile) =>
            {
                try
                {
                    if (monitor.cancelled) return (tile.Item1, tile.Item2, null, null, null, null, null);

                    Mat[] beforeImgs = new Mat[9];
                    Mat[] beforeUDMImgs = new Mat[9];
                    Mat[] afterImgs = new Mat[9];
                    Mat[] afterUDMImgs = new Mat[9];

                    Envelope envelope = null;

                    var (sx, sy) = tile;

                    int idx = 0;

                    for (int y = sy + 1; y >= sy - 1; y--)
                    {
                        for (int x = sx - 1; x <= sx + 1; x++)
                        {
                            var watch = System.Diagnostics.Stopwatch.StartNew();

                            var (beforeImg, beforeCCImg, envel) = await planetReader.FindImage(x, y, bMonth, bYear, savePath + "/before");
                            if (beforeImg == null) return (x, y, null, null, null, null, null);

                            var (afterImg, afterUDMImg, _) = await planetReader.FindImage(x, y, aMonth, aYear, savePath + "/after");
                            if (afterImg == null) return (x, y, null, null, null, null, null);

                            watch.Stop();
                            Console.WriteLine($"Tile {x}, {y} downloaded in {watch.ElapsedMilliseconds} ms");

                            if (idx == 4) envelope = envel;

                            beforeImgs[idx] = beforeImg;
                            beforeUDMImgs[idx] = beforeCCImg;
                            afterImgs[idx] = afterImg;
                            afterUDMImgs[idx] = afterUDMImg;
                            idx++;
                        }
                    }

                    return (sx, sy, beforeImgs, beforeUDMImgs, envelope, afterImgs, afterUDMImgs);
                }
                catch (Exception ex)
                {
                    System.Windows.MessageBox.Show("Error in Download Block:\n\n" + ex.Message);
                }

                return (tile.Item1, tile.Item2, null, null, null, null, null);
            },
            new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = 1,
                BoundedCapacity = 1,
                EnsureOrdered = false
            });


            var preprocessBlock = new TransformBlock<(int, int, Mat[], Mat[], Envelope, Mat[], Mat[]), (int, int, Mat, Mat, Envelope, Mat, Mat)>((packet) =>
            {
                try
                {
                    var watch = System.Diagnostics.Stopwatch.StartNew();

                    var (x, y, beforeImgs, beforeUDMImgs, envelope, afterImgs, afterUDMImgs) = packet;

                    if (monitor.cancelled || beforeImgs == null) return (x, y, null, null, null, null, null);

                    Mat beforeImg = MergeGrid3x3(beforeImgs);
                    Mat beforeCCImg = MergeGrid3x3(beforeUDMImgs);
                    Mat afterImg = MergeGrid3x3(afterImgs);
                    Mat afterCCImg = MergeGrid3x3(afterUDMImgs);

                    Cv2.MedianBlur(beforeImg, beforeImg, 5);
                    Cv2.MedianBlur(afterImg, afterImg, 5);

                    //extend envelope by 1.25x to match the merged image
                    double xCenter = (envelope.XMin + envelope.XMax) / 2;
                    double yCenter = (envelope.YMin + envelope.YMax) / 2;
                    double xHalfWidth = (envelope.XMax - envelope.XMin) / 2 * 1.25;
                    double yHalfHeight = (envelope.YMax - envelope.YMin) / 2 * 1.25;
                    Envelope mergedEnvelope = EnvelopeBuilderEx.CreateEnvelope(xCenter - xHalfWidth, yCenter - yHalfHeight, xCenter + xHalfWidth, yCenter + yHalfHeight, envelope.SpatialReference);

                    watch.Stop();
                    Console.WriteLine($"Tile {x}, {y} preprocessed in {watch.ElapsedMilliseconds} ms");

                    return (x, y, beforeImg, beforeCCImg, envelope, afterImg, afterCCImg);
                }
                catch (Exception ex)
                {
                    System.Windows.MessageBox.Show("Error in Pre Processing Block:\n\n" + ex.Message);
                }

                return (packet.Item1, packet.Item2, null, null, null, null, null);
            },
            new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = 1,
                BoundedCapacity = 1,
                EnsureOrdered = false
            });

            var predictionBlock = new TransformBlock<(int, int, Mat, Mat, Envelope, Mat, Mat), (int, int, Mat, Mat, Envelope, Mat, Mat, ByteVector, ByteVector)>((packet) =>
            {
                try
                {
                    var watch = System.Diagnostics.Stopwatch.StartNew();

                    var (x, y, beforeImg, beforeCCImg, envelope, afterImg, afterCCImg) = packet;

                    if (monitor.cancelled || beforeImg == null || afterImg == null) return (x, y, null, null, null, null, null, null, null);

                    ByteVector tornadoPredictionA = tpp.analyze(beforeImg, afterImg, beforeImg.Width, beforeImg.Height);
                    ByteVector tornadoPredictionB = tpp64_256.analyze(beforeImg, afterImg, beforeImg.Width, beforeImg.Height);

                    watch.Stop();
                    Console.WriteLine($"Tile {x}, {y} predicted in {watch.ElapsedMilliseconds} ms");

                    return (x, y, beforeImg, beforeCCImg, envelope, afterImg, afterCCImg, tornadoPredictionA, tornadoPredictionB);
                }
                catch (Exception ex)
                {
                    System.Windows.MessageBox.Show("Error in Prediction Block:\n\n" + ex.Message);
                }

                return (packet.Item1, packet.Item2, null, null, null, null, null, null, null);
            }, 
            new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = 1,
                BoundedCapacity = 1,
                EnsureOrdered = false
            });

            var postProcessBlock = new TransformBlock<(int, int, Mat, Mat, Envelope, Mat, Mat, ByteVector, ByteVector), (int, int, int, int, Mat, Mat, Mat, Mat)> (async (packet) =>
            {
                try 
                { 
                    if (monitor.cancelled) return (packet.Item1, packet.Item2, 0, 0, null, null, null, null);
                    var watch = System.Diagnostics.Stopwatch.StartNew();

                    var (x, y, beforeImg, beforeCCImg, envelope, afterImg, afterCCImg, tornadoPredictionA, tornadoPredictionB) = packet;

                    if (beforeImg == null || afterImg == null) return (x, y, 0, 0, null, null, null, null);

                    Cv2.ImWrite("C:/Users/danie/Documents/Experiments/Satellite/patch_predictor/testing_events/Lac des Deux Cantons/before.png", beforeImg);
                    Cv2.ImWrite("C:/Users/danie/Documents/Experiments/Satellite/patch_predictor/testing_events/Lac des Deux Cantons/after.png", afterImg);

                    Mat predAImg = ByteVector.ToMat(tornadoPredictionA, beforeImg.Size());
                    Cv2.ImWrite("C:/Users/danie/Documents/Experiments/Satellite/patch_predictor/testing_events/Lac des Deux Cantons/predA_" + x + "_" + y + ".png", predAImg);
                    //Mat predASmall = new();
                    //Cv2.Resize(predAImg, predASmall, new OpenCvSharp.Size(beforeImg.Width / 4, beforeImg.Height / 4), interpolation: InterpolationFlags.Area);

                    //Cv2.ImShow("PredA", predASmall);
                    //Cv2.WaitKey(0);
                    //Cv2.DestroyAllWindows();

                    Mat predBImg = ByteVector.ToMat(tornadoPredictionB, beforeImg.Size());
                    Cv2.ImWrite("C:/Users/danie/Documents/Experiments/Satellite/patch_predictor/testing_events/Lac des Deux Cantons/predB_" + x + "_" + y + ".png", predBImg);
                    //Mat predBSmall = new();
                    //Cv2.Resize(predBImg, predBSmall, new OpenCvSharp.Size(beforeImg.Width / 4, beforeImg.Height / 4), interpolation: InterpolationFlags.Area);

                    //Cv2.ImShow("PredB", predBSmall);
                    //Cv2.WaitKey(0);
                    //Cv2.DestroyAllWindows();

                    Mat predImg = FloatMulNormalized(predAImg, predBImg);
                    //Mat predSmall = new();
                    //Cv2.Resize(predImg, predSmall, new OpenCvSharp.Size(beforeImg.Width / 4, beforeImg.Height / 4), interpolation: InterpolationFlags.Area);

                    //Cv2.ImShow("Pred", predSmall);
                    //Cv2.WaitKey(0);
                    //Cv2.DestroyAllWindows();

                    Mat landcoverImg = await LandCoverMask(envelope, beforeImg.Size());
                    //Mat landcoverSmall = new();
                    //Cv2.Resize(landcoverImg, landcoverSmall, new OpenCvSharp.Size(beforeImg.Width / 4, beforeImg.Height / 4), interpolation: InterpolationFlags.Area);

                    //Cv2.ImShow("Landcover Mask", landcoverSmall);
                    //Cv2.WaitKey(0);
                    //Cv2.DestroyAllWindows();

                    Mat colorChangeMask = ColorChangeMask(beforeImg, afterImg);
                    Mat colorChangeSmall = new();
                    //Cv2.Resize(colorChangeMask, colorChangeSmall, new OpenCvSharp.Size(beforeImg.Width / 4, beforeImg.Height / 4), interpolation: InterpolationFlags.Area);

                    //Cv2.ImShow("Color Change Mask", colorChangeSmall);
                    //Cv2.WaitKey(0);
                    //Cv2.DestroyAllWindows();

                    Mat mask = new();
                    Cv2.BitwiseAnd(colorChangeMask, landcoverImg, mask);
                    Cv2.BitwiseAnd(mask, beforeCCImg, mask);
                    Cv2.BitwiseAnd(mask, afterCCImg, mask);

                    Mat diffImg = AbsDifferenceImage(beforeImg, afterImg);
                    Cv2.BitwiseAnd(mask, diffImg, diffImg);
                    Cv2.ImWrite("C:/Users/danie/Documents/Experiments/Satellite/patch_predictor/testing_events/Lac des Deux Cantons/diff_" + x + "_" + y + ".png", diffImg);

                    //Mat diffSmall = new();
                    //Cv2.Resize(diffImg, diffSmall, new OpenCvSharp.Size(beforeImg.Width / 4, beforeImg.Height / 4), interpolation: InterpolationFlags.Area);

                    //Cv2.ImShow("Difference Image", diffSmall);
                    //Cv2.WaitKey(0);
                    //Cv2.DestroyAllWindows();

                    Cv2.CvtColor(mask, mask, ColorConversionCodes.BGR2GRAY);

                    Cv2.BitwiseAnd(predImg, mask, predImg);
                    Cv2.ImWrite("C:/Users/danie/Documents/Experiments/Satellite/patch_predictor/testing_events/Lac des Deux Cantons/pred_masked_" + x + "_" + y + ".png", predImg);
                    //Mat predSmallMasked = new();
                    //Cv2.Resize(predImg, predSmallMasked, new OpenCvSharp.Size(beforeImg.Width / 4, beforeImg.Height / 4), interpolation: InterpolationFlags.Area);

                    //Cv2.ImShow("Masked Prediction", predSmallMasked);
                    //Cv2.WaitKey(0);
                    //Cv2.DestroyAllWindows();

                    //Cv2.MorphologyEx(predImg, predImg, MorphTypes.Open, Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(15, 15)));
                    Cv2.MinMaxLoc(predImg, out double _, out double maxVal);

                    int pxCount = Cv2.CountNonZero(predImg.Threshold(Math.Max(maxVal-1, 0), 255, ThresholdTypes.Binary));

                    predImg = predImg.Threshold(127, 255, ThresholdTypes.Binary);

                    watch.Stop();
                    Console.WriteLine($"Tile {x}, {y} post processed in {watch.ElapsedMilliseconds} ms");

                    return (x, y, (int)maxVal, pxCount, predImg, diffImg, beforeImg, afterImg);
                }
                catch (Exception ex) {
                    System.Windows.MessageBox.Show( "Error in Post Processing Block:\n\n" + ex.Message );
                }

                return (packet.Item1, packet.Item2, 0, 0, null, null, null, null);
            },
            new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = 1,
                BoundedCapacity = 1,
                EnsureOrdered = false
            });

            var saveBlock = new ActionBlock<(int, int, int, int, Mat, Mat, Mat, Mat)>(async (packet) =>
            {
                try
                {
                    if (monitor.cancelled) return;

                    var (x, y, maxVal, pxCount, predImg, diffImg, beforeImg, afterImg) = packet;

                    if (maxVal < 200)
                    {
                        Console.WriteLine($"Tile {x}, {y} skipped");
                        monitor.Update();
                        return;
                    }

                    var watch = System.Diagnostics.Stopwatch.StartNew();

                    Cv2.ImWrite(savePath + $"/diff_{x}_{y}.png", diffImg);
                    Cv2.ImWrite(savePath + $"/before_{x}_{y}.png", beforeImg);
                    Cv2.ImWrite(savePath + $"/after_{x}_{y}.png", afterImg);
                    byte[] afterBytes = File.ReadAllBytes(savePath + $"/after_{aYear}_{aMonth}_{x}_{y}.bin");

                    Tiff pred = GeoTiff.CreateFromReference(afterBytes, savePath + $"\\pred_{x}_{y}.tif");
                    Mat predRoi = new(predImg, new Rect(512, 512, 4096, 4096));
                    Mat predCropped = new();
                    predRoi.CopyTo(predCropped);

                    GeoTiff.WriteImage(pred, predCropped);

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
                MaxDegreeOfParallelism = 1,
                BoundedCapacity = 1,
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

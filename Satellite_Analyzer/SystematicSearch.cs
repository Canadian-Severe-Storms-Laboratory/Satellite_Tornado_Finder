using ArcGIS.Core.CIM;
using ArcGIS.Core.Data.Raster;
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
using System.Runtime.InteropServices;
using System.Collections.Concurrent;
using System.Linq;
using Newtonsoft.Json;

namespace Satellite_Analyzer
{
    public struct SearchResult(int x, int y, int pc)
    {
        public int tileX = x, tileY = y, pixelCount = pc;

        public override readonly string ToString()
        {
            return string.Format("{0}, {1} - {2}", tileX, tileY, pixelCount);
        }
    }

    public class SystematicSearch
    {
        private static string CreateResultsFolder()
        {
            string path = GetProjectPath() + "\\SatelliteAnalysis";
            if (!Directory.Exists(path)) Directory.CreateDirectory(path);

            string dateTime = System.DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss");

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
        private static List<(int, int)> tiles;
        private static ConcurrentBag<SearchResult> significantTiles;
        private static string savePath;
        private static GroupLayer predGroup;
        private static GroupLayer diffGroup;
        private static Monitor monitor;

        private static async Task InitalizeSearch(Polygon polygon)
        {
            OpenConsole();

            planetReader = new();
            await planetReader.BuildBaseMapDict();

            await LandCover.Initalize();

            tiles = PolygonToTiles(polygon);

            significantTiles = [];

            tpp = new(AddinAssemblyLocation() + "\\tornado_patch_predictor_de_norm.onnx");

            savePath = CreateResultsFolder();

            predGroup = await QueuedTask.Run(() => { return LayerFactory.Instance.CreateGroupLayer(MapView.Active.Map, 0, "Tornado_Prediction"); });
            diffGroup = await QueuedTask.Run(() => { return LayerFactory.Instance.CreateGroupLayer(MapView.Active.Map, 0, "Differnce_Images"); });

            monitor = new("Search Progress", tiles.Count);
        }

        public static async Task<List<SearchResult>> Search(Polygon polygon, int bMonth, int bYear, int aMonth, int aYear)
        {
            await InitalizeSearch(polygon);

            monitor.Start();

            var downloadBlock = new TransformBlock<(int, int), (int, int, byte[], byte[], int, Envelope, byte[], byte[], int)>(async (tile) =>
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
            });

            var preprocessBlock = new TransformBlock<(int, int, byte[], byte[], int, Envelope, byte[], byte[], int), (int, int, Mat, Mat, byte[], Envelope, Mat, Mat, byte[])>((packet) =>
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
            });

            var predictionBlock = new TransformBlock<(int, int, Mat, Mat, byte[], Envelope, Mat, Mat, byte[]), (int, int, Mat, Mat, byte[], Envelope, Mat, Mat, byte[], ByteVector)>((packet) =>
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();

                var (x, y, beforeImg, beforeCCImg, beforeBytes, envelope, afterImg, afterCCImg, afterBytes) = packet;

                if (monitor.cancelled || beforeImg == null || afterImg == null) return (x, y, null, null, null, null, null, null, null, null);

                ByteVector tornadoPrediction = tpp.analyze(beforeImg, afterImg, beforeImg.Width, beforeImg.Height);

                watch.Stop();
                Console.WriteLine($"Tile {x}, {y} predicted in {watch.ElapsedMilliseconds} ms");

                return (x, y, beforeImg, beforeCCImg, beforeBytes, envelope, afterImg, afterCCImg, afterBytes, tornadoPrediction);

            });

            var postProcessBlock = new TransformBlock<(int, int, Mat, Mat, byte[], Envelope, Mat, Mat, byte[], ByteVector), (int, int, int, Mat, Mat, byte[], byte[])> (async (packet) =>
            {
                if (monitor.cancelled) return (packet.Item1, packet.Item2, 0, null, null, null, null);
                var watch = System.Diagnostics.Stopwatch.StartNew();

                var (x, y, beforeImg, beforeCCImg, beforeBytes, envelope, afterImg, afterCCImg, afterBytes, tornadoPrediction) = packet;

                if (beforeImg == null || afterImg == null) return (x, y, 0, null, null, null, null);

                Mat predImg = ByteVector.ToMat(tornadoPrediction, beforeImg.Size());

                Mat landcoverImg = await LandCoverMask(envelope, beforeImg.Size());

                Mat mask = new();
                Cv2.BitwiseAnd(afterCCImg, landcoverImg, mask);
                Cv2.BitwiseAnd(mask, beforeCCImg, mask);

                Mat diffImg = AbsDifferenceImage(beforeImg, afterImg);
                Cv2.BitwiseAnd(mask, diffImg, diffImg);

                Cv2.CvtColor(mask, mask, ColorConversionCodes.BGR2GRAY);

                Cv2.BitwiseAnd(predImg, mask, predImg);
                Cv2.MorphologyEx(predImg, predImg, MorphTypes.Open, Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(25, 25)));

                int pxCount = Cv2.CountNonZero(predImg);

                watch.Stop();
                Console.WriteLine($"Tile {x}, {y} post processed in {watch.ElapsedMilliseconds} ms");

                return (x, y, pxCount, predImg, diffImg, beforeBytes, afterBytes);
            });

            var saveBlock = new ActionBlock<(int, int, int, Mat, Mat, byte[], byte[])>(async (packet) =>
            {
                if (monitor.cancelled) return;

                var (x, y, pxCount, predImg, diffImg, beforeBytes, afterBytes) = packet;

                string imgName = $"_{x}_{y}.png";

                if (pxCount < 2000)
                { 
                    Console.WriteLine($"Tile {x}, {y} skipped");
                    monitor.Update();
                    return;
                }

                var watch = System.Diagnostics.Stopwatch.StartNew();

                await File.WriteAllBytesAsync(savePath + "\\before" + imgName, beforeBytes);
                await File.WriteAllBytesAsync(savePath + "\\after" + imgName, afterBytes);

                significantTiles.Add(new(x, y, pxCount));

                Mat diffMask = new();
                Cv2.InRange(diffImg, new Scalar(1, 1, 1), new Scalar(255, 255, 255), diffMask);

                Mat[] diffchannels = Cv2.Split(diffImg);

                Cv2.Merge([diffchannels[2], diffchannels[1], diffchannels[0], diffMask], diffImg);

                await QueuedTask.Run(() =>
                {
                    var rd = DuplicateRasterDataset(savePath, "before" + imgName, savePath, $"pred_{x}_{y}.tif");
                    File.Copy(savePath + $"\\pred_{x}_{y}.tif", savePath + $"\\diff_{x}_{y}.tif", true);

                    Raster raster = rd.CreateFullRaster();

                    WriteByteRaster(predImg, raster, channels: 4);

                    var layer = LoadRasterLayer(savePath, $"pred_{x}_{y}.tif", predGroup);

                    CIMRasterRGBColorizer colorizer = new()
                    {
                        AlphaBandIndex = 3,
                        StretchType = RasterStretchType.MinimumMaximum,
                        UseAlphaBand = true,
                        UseGreenBand = false,
                        UseBlueBand = false
                    };

                    layer.SetColorizer(colorizer);

                    //rd = DuplicateRasterDataset(savePath, "before" + imgName, savePath, $"diff_{x}_{y}.tif");
                    rd = OpenRasterDataset(savePath, $"diff_{x}_{y}.tif");
                    raster = rd.CreateFullRaster();

                    WriteRaster<byte>(diffImg, raster);

                    layer = LoadRasterLayer(savePath, $"diff_{x}_{y}.tif", diffGroup);

                    colorizer = new()
                    {
                        AlphaBandIndex = 3,
                        StretchType = RasterStretchType.MinimumMaximum,
                        UseAlphaBand = true,
                    };

                    layer.SetColorizer(colorizer);
                });

                monitor.Update();
                watch.Stop();
                Console.WriteLine($"Tile {x}, {y} saved in {watch.ElapsedMilliseconds} ms");
            },
            new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = 4
            });

            downloadBlock.LinkTo(preprocessBlock, new DataflowLinkOptions() { PropagateCompletion = true });
            preprocessBlock.LinkTo(predictionBlock, new DataflowLinkOptions() { PropagateCompletion = true });
            predictionBlock.LinkTo(postProcessBlock, new DataflowLinkOptions() { PropagateCompletion = true });
            postProcessBlock.LinkTo(saveBlock, new DataflowLinkOptions() { PropagateCompletion = true });

            foreach (var tile in tiles) await downloadBlock.SendAsync(tile);

            downloadBlock.Complete();

            await saveBlock.Completion;

            monitor.Stop();

            SaveSearchResults();

            return [.. significantTiles];
        }

        private static void SaveSearchResults()
        {
            string path = savePath + "\\results.json";

            Dictionary<string, object> results = new()
            {
                { "folderPath", savePath },
                { "tiles", significantTiles }
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

        public static Mat AbsDifferenceImage(Mat before, Mat after)
        {
            Mat diffImg = new();

            Cv2.Absdiff(after, before, diffImg);

            //adjust brightness and contrast
            diffImg.ConvertTo(diffImg, -1, 30, -300);

            return diffImg;
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

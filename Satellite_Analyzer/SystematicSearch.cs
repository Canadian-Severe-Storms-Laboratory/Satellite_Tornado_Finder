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
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;
using System.Linq;

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

        private static PlanetReader planetReader;
        private static TornadoPatchPredictor tpp;
        private static List<(int, int)> tiles;
        private static ConcurrentBag<SearchResult> significantTiles;
        private static string savePath;
        private static GroupLayer group;
        private static Monitor monitor;
        private static int bMonth, bYear, aMonth, aYear;

        private static async Task InitalizeSearch(Polygon polygon, int bM, int bY, int aM, int aY)
        {
            OpenConsole();

            bMonth = bM;
            bYear = bY;
            aMonth = aM;
            aYear = aY;

            planetReader = new();
            await planetReader.BuildBaseMapDict();

            tiles = PolygonToTiles(polygon);

            significantTiles = [];

            tpp = new(AddinAssemblyLocation() + "\\tornado_patch_predictor_de_norm.onnx");

            savePath = CreateResultsFolder();

            group = await QueuedTask.Run(() => { return LayerFactory.Instance.CreateGroupLayer(MapView.Active.Map, 0, "Tornado_Prediction"); });

            monitor = new("Search Progress", tiles.Count);
        }

        public static async Task<List<SearchResult>> Search(Polygon polygon, int bMonth, int bYear, int aMonth, int aYear)
        {
            await InitalizeSearch(polygon, bMonth, bYear, aMonth, aYear);

            monitor.Start();

            var downloadBlock = new TransformBlock<(int, int), (int, int, Mat, Mat, Envelope, Mat, Mat)>(async (tile) =>
            {
                if (monitor.cancelled) return (tile.Item1, tile.Item2, null, null, null, null, null);

                var watch = System.Diagnostics.Stopwatch.StartNew();
                var (x, y) = tile;

                string imgName = $"tile_{x}_{y}.png";
                string imgPath = savePath + "\\" + imgName;

                var (beforeImg, beforeCCImg, envelope) = await planetReader.FindImage(x, y, bMonth, bYear, imgPath);
                if (beforeImg == null) return (x, y, null, null, null, null, null);

                var (afterImg, afterCCImg, _) = await planetReader.FindImage(x, y, aMonth, aYear);
                if (afterImg == null) return (x, y, null, null, null, null, null);

                watch.Stop();
                Console.WriteLine($"Tile {x}, {y} downloaded in {watch.ElapsedMilliseconds} ms");

                return (x, y, beforeImg, beforeCCImg, envelope, afterImg, afterCCImg);
            });

            var predictionBlock = new TransformBlock<(int, int, Mat, Mat, Envelope, Mat, Mat), (int, int, int, Mat)>(async (packet) =>
            {
                if (monitor.cancelled) return (packet.Item1, packet.Item2, 0, null);

                var watch = System.Diagnostics.Stopwatch.StartNew();
                var (x, y, beforeImg, beforeCCImg, envelope, afterImg, afterCCImg) = packet;

                if (beforeImg == null || afterImg == null) return (x, y, 0, null);

                Cv2.MedianBlur(beforeImg, beforeImg, 7);
                Cv2.MedianBlur(afterImg, afterImg, 7);

                ByteVector tornadoPrediction = tpp.analyze(beforeImg, afterImg, beforeImg.Width, beforeImg.Height);
                Mat predImg = ByteVector.ToMat(tornadoPrediction, beforeImg.Size());

                Mat landcoverImg = await LandCoverMask(envelope, beforeImg.Size());

                Mat mask = new();
                Cv2.BitwiseAnd(afterCCImg, landcoverImg, mask);
                Cv2.BitwiseAnd(mask, beforeCCImg, mask);

                Cv2.CvtColor(mask, mask, ColorConversionCodes.BGR2GRAY);

                Cv2.BitwiseAnd(predImg, mask, predImg);
                Cv2.MorphologyEx(predImg, predImg, MorphTypes.Open, Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(25, 25)));

                int pxCount = Cv2.CountNonZero(predImg);

                watch.Stop();
                Console.WriteLine($"Tile {x}, {y} predicted in {watch.ElapsedMilliseconds} ms");

                return (x, y, pxCount, predImg);
            });

            var saveBlock = new ActionBlock<(int, int, int, Mat)>(async (packet) =>
            {
                if (monitor.cancelled) return;

                var (x, y, pxCount, predImg) = packet;

                if (pxCount < 2000)
                {
                    Console.WriteLine($"Tile {x}, {y} skipped");
                    monitor.Update();
                    return;
                }

                var watch = System.Diagnostics.Stopwatch.StartNew();

                significantTiles.Add(new(x, y, pxCount));

                string imgName = $"tile_{x}_{y}.png";

                await QueuedTask.Run(() =>
                {
                    var rd = DuplicateRasterDataset(savePath, imgName, savePath, $"pred_{x}_{y}.tif");
                    Raster raster = rd.CreateFullRaster();

                    WriteByteRaster(predImg, raster, channels: 4);

                    var layer = LoadRasterLayer(savePath, $"pred_{x}_{y}.tif", group);

                    CIMRasterRGBColorizer colorizer = new()
                    {
                        AlphaBandIndex = 3,
                        StretchType = RasterStretchType.MinimumMaximum,
                        UseAlphaBand = true,
                        UseGreenBand = false,
                        UseBlueBand = false
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

            downloadBlock.LinkTo(predictionBlock, new DataflowLinkOptions() { PropagateCompletion = true });
            predictionBlock.LinkTo(saveBlock, new DataflowLinkOptions() { PropagateCompletion = true });

            foreach (var tile in tiles) await downloadBlock.SendAsync(tile);

            downloadBlock.Complete();

            await saveBlock.Completion;

            monitor.Stop();

            return [.. significantTiles];
        }

        static async Task<Mat> LandCoverMask(Envelope envelope, Size size)
        {
            Mat landcoverImg = LandCover.TypeMask(await LandCover.GetSection(envelope), LandCover.potentialForestTypes);

            Cv2.Resize(landcoverImg, landcoverImg, size, interpolation: InterpolationFlags.Cubic);
            Cv2.Threshold(landcoverImg, landcoverImg, 127, 255, ThresholdTypes.Binary);
            Cv2.Erode(landcoverImg, landcoverImg, Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(7, 7)), iterations: 2);
            Cv2.CvtColor(landcoverImg, landcoverImg, ColorConversionCodes.GRAY2BGR);

            return landcoverImg;
        }


        static List<(int, int)> PolygonToTiles(Polygon polygon)
        {
            List<(int, int)> tiles = [];

            //Envelope polygonEnvelope = (Envelope)GeometryEngine.Instance.Project(polygon, SpatialReferences.WebMercator);

            Envelope polygonEnvelope = polygon.Extent;

            var (xmin, ymin) = PlanetReader.TileIndex(polygonEnvelope.YMin, -polygonEnvelope.XMin, mercator: true);
            var (xmax, ymax) = PlanetReader.TileIndex(polygonEnvelope.YMax, -polygonEnvelope.XMax, mercator: true);

            for (int y = (int)ymax + 1; y >= (int)ymin; y--)
            {
                for (int x = (int)xmin; x <= (int)xmax + 1; x++)
                {
                    Envelope tileEnvelope = PlanetReader.TileIndexToEnvelope(x, y);

                    if (GeometryEngine.Instance.Intersects(tileEnvelope, polygon)) tiles.Add((x, y));
                }
            }

            return tiles;
        }

        /*
           foreach (var (x, y) in tiles)
            {
                monitor.Update();
                if (monitor.cancelled) break;

                string imgName = $"tile_{x}_{y}.png";
                string imgPath = savePath + "\\" + imgName;

                var (beforeImg, beforeCCImg, envelope) = await planetReader.FindImage(x, y, bMonth, bYear, imgPath);

                if (beforeImg == null) continue;    
                if (monitor.cancelled) break;

                var (afterImg, afterCCImg, _) = await planetReader.FindImage(x, y, aMonth, aYear);

                if (afterImg == null) continue;
                if (monitor.cancelled) break;

                Cv2.MedianBlur(beforeImg, beforeImg, 7);
                Cv2.MedianBlur(afterImg, afterImg, 7);

                if (monitor.cancelled) break;

                ByteVector tornadoPrediction = tpp.analyze(beforeImg, afterImg, beforeImg.Width, beforeImg.Height);
                Mat predImg = ByteVector.ToMat(tornadoPrediction, beforeImg.Size());

                if (monitor.cancelled) break;

                Mat landcoverImg = await LandCoverMask(envelope, beforeImg.Size());

                Mat mask = new();
                Cv2.BitwiseAnd(afterCCImg, landcoverImg, mask);
                Cv2.BitwiseAnd(mask, beforeCCImg, mask);

                //fix
                Cv2.CvtColor(mask, mask, ColorConversionCodes.BGR2GRAY);

                Cv2.BitwiseAnd(predImg, mask, predImg);
                Cv2.MorphologyEx(predImg, predImg, MorphTypes.Open, Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(21, 21)));
                //Cv2.Erode(predImg, predImg, Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(17, 17)));

                if (monitor.cancelled) break;

                int pxCount = Cv2.CountNonZero(predImg);

                //if (pxCount < 2000) continue;

                significantTiles.Add(new (x, y, pxCount));

                //Cv2.CvtColor(predImg, predImg, ColorConversionCodes.GRAY2BGR);
                Cv2.Merge([predImg, predImg, predImg, predImg], predImg);

                await QueuedTask.Run(() => {
                    var rd = DuplicateRasterDataset(savePath, imgName, savePath, $"pred_{x}_{y}.tif");
                    Raster raster = rd.CreateFullRaster();
                    WriteRaster<byte>(predImg, raster);

                    var layer = LoadRasterLayer(savePath, $"pred_{x}_{y}.tif", group);

                    CIMRasterRGBColorizer colorizer = new()
                    {
                        AlphaBandIndex = 3,
                        StretchType = RasterStretchType.MinimumMaximum,
                        UseAlphaBand = true,
                        UseGreenBand = false,
                        UseBlueBand = false
                    };

                    layer.SetColorizer(colorizer);

                });

            }
         */

    }
}

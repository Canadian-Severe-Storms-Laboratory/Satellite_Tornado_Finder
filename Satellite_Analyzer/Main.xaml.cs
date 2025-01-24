using OpenCvSharp;
using OpenCvSharp.Extensions;
using ScottPlot;
using System;
using System.Drawing;
using System.Windows;
using static ArcGISUtils.Utils;
using ScottPlot.Plottables;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Core.Geometry;
using ArcGIS.Desktop.Mapping;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using CsvHelper;
using System.Windows.Input;
using ArcGIS.Desktop.Internal.Mapping;
using System.Windows.Media;
using OpenTK.Windowing.Common.Input;
using ArcGIS.Desktop.Internal.Mapping.Locate;

namespace Satellite_Analyzer
{

    public partial class Main : ArcGIS.Desktop.Framework.Controls.ProWindow
    {
        private bool loaded = false;
        private PlanetReader planetReader = new PlanetReader();

        private List<ImageRect> imgPlottables = [];

        private Mat beforeImg = null;
        private Mat afterImg = null;
        private Mat landcoverImg = null;
        private Mat beforeCCImg = null;
        private Mat afterCCImg = null;
        private Mat beforeUDMImg = null;
        private Mat afterUDMImg = null;
        private Mat diffImg = null;
        private Mat predImg = null;

        private Envelope envelope = null;
        private (double, double) worstPoint;
        private Random r = new Random();

        private TornadoPatchPredictor tpp;

        public Main()
        {
            InitializeComponent();

            LoadSavedEvents();

        }

        private void LoadSavedEvents()
        {
            //update to relative path...
            var filePath = AddinAssemblyLocation() + "\\forest_tornadoes_modified.csv";

            var culture = new System.Globalization.CultureInfo("en-US", false);
            culture.NumberFormat.NumberDecimalDigits = 4;
            culture.NumberFormat.CurrencyDecimalDigits = 4;
            culture.NumberFormat.PercentDecimalDigits = 4;

            using var reader = new StreamReader(filePath);
            using var csv = new CsvReader(reader, culture);
            List<SevereStorm> events = csv.GetRecords<SevereStorm>().ToList();
            events.Sort((a, b) => a.location.CompareTo(b.location));

            eventList.ItemsSource = events;
        }

        private async void Search(object sender=null, RoutedEventArgs e=null)
        {
            if (!loaded) return;

            loadingLabel.Visibility = Visibility.Visible;

            var (bMonth, bYear) = beforeDate.GetDate();
            var (aMonth, aYear) = afterDate.GetDate();

            if (tileSearchInput.IsTileIndexSearch()) 
            {
                var (tileX, tileY) = tileSearchInput.GetTileIndex();

                var (beforeBytes, beforeMaskBytes, envel, beforeMaskType) = await planetReader.FindImage(tileX, tileY, bMonth, bYear);

                beforeImg = Cv2.ImDecode(beforeBytes, ImreadModes.Color);
                beforeCCImg = PlanetReader.DecodeUDM(beforeMaskBytes, beforeMaskType);
                envelope = envel;

                var (afterBytes, afterMaskBytes, _, afterMaskType) = await planetReader.FindImage(tileX, tileY, aMonth, aYear);

                afterImg = Cv2.ImDecode(afterBytes, ImreadModes.Color);
                afterCCImg = PlanetReader.DecodeUDM(afterMaskBytes, afterMaskType);
            }
            else
            {
                var (lat, lon) = tileSearchInput.GetCoordinates();

                (beforeImg, beforeCCImg, envelope, worstPoint) = await planetReader.FindImage(lat, lon, bMonth, bYear);
                (afterImg, afterCCImg, _, _) = await planetReader.FindImage(lat, lon, aMonth, aYear);
            }

            if (beforeImg == null || afterImg == null)
            {
                loadingLabel.Visibility = Visibility.Hidden;
                MessageBox.Show("Failed to load image from Planet");
                return;
            }

            Cv2.MedianBlur(beforeImg, beforeImg, 7);
            Cv2.MedianBlur(afterImg, afterImg, 7);

            ByteVector tornadoPrediction = tpp.analyze(beforeImg, afterImg, beforeImg.Width, beforeImg.Height);
            predImg = ByteVector.ToMat(tornadoPrediction, beforeImg.Size());

            landcoverImg = await SystematicSearch.LandCoverMask(envelope, beforeImg.Size());

            Mat mask = new();
            beforeUDMImg = new Mat();
            afterUDMImg = new Mat();

            Cv2.BitwiseAnd(afterCCImg, landcoverImg, mask);
            Cv2.BitwiseAnd(mask, beforeCCImg, mask);

            Cv2.BitwiseAnd(beforeImg, mask, beforeUDMImg);
            Cv2.BitwiseAnd(afterImg, mask, afterUDMImg);

            diffImg = SystematicSearch.AbsDifferenceImage(beforeUDMImg, afterUDMImg);

            imgPlottables = [MatToImageRect(beforeImg), MatToImageRect(afterImg), MatToImageRect(landcoverImg),
                             MatToImageRect(beforeCCImg), MatToImageRect(afterCCImg), MatToImageRect(beforeUDMImg), 
                             MatToImageRect(afterUDMImg), MatToImageRect(diffImg), MatToImageRect(predImg)];

            rects.Clear();
            otherRects.Clear();

            UpdatePlot();
            mainPlot.Plot.Axes.AutoScale();
        }

        private void UpdatePlot(object sender = null, RoutedEventArgs e = null)
        {
            if (!loaded || beforeImg == null) return;

            var plt = mainPlot.Plot;
            plt.Clear();

            var imRect = imgPlottables[imageList.SelectedIndex];

            plt.PlottableList.Add(imRect);
            plt.Add.Marker(worstPoint.Item1, worstPoint.Item2, shape: MarkerShape.OpenTriangleUp, color: ScottPlot.Color.FromARGB(0xFF01F9C6), size: 50);
            plt.PlottableList.AddRange(rects);

            //plt.Axes.AutoScale();

            loadingLabel.Visibility = Visibility.Hidden;

            mainPlot.Refresh();
        }

        public static ImageRect MatToImageRect(Mat image)
        {
            return new ImageRect
            {
                Image = new(BitmapToBytes(BitmapConverter.ToBitmap(image))),
                Rect = new(0, image.Cols, 0, image.Rows)
            };
        }

        public static byte[] BitmapToBytes(Bitmap img)
        {
            ImageConverter converter = new ImageConverter();
            return (byte[])converter.ConvertTo(img, typeof(byte[]));
        }

        private async void WindowLoaded(object sender, RoutedEventArgs e)
        {
            loadingLabel.Visibility = Visibility.Visible;

            PreLoadDlls();
            await LandCover.Initalize();
            await planetReader.BuildBaseMapDict();

            var plt = mainPlot.Plot;
            plt.Axes.SquareUnits();
            plt.Layout.Frameless();
            plt.DataBackground.Color = ScottPlot.Colors.Black;
            plt.HideAxesAndGrid();

            mainPlot.Refresh();

            loadingLabel.Visibility = Visibility.Hidden;

            //update to relative path...
            tpp = new(AddinAssemblyLocation() + "\\tornado_patch_predictor_de_norm.onnx");
            
            if (tpp.usingGPU)
            {
                executionProviderLabel.Content = "Using GPU";
                executionProviderLabel.Foreground = new SolidColorBrush(System.Windows.Media.Colors.Green);
            }
            else
            {
                executionProviderLabel.Content = "Using CPU";
                executionProviderLabel.Foreground = new SolidColorBrush(System.Windows.Media.Colors.Red);
            }

            loaded = true;
        }

        private IEnumerable<RasterLayer> GetRasterLayers()
        {
            //get current map
            var map = MapView.Active.Map;

            //get all raster layers on map
            return map.GetLayersAsFlattenedList().OfType<RasterLayer>();
        }

        private void UpdateSearchParams(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            SevereStorm storm = eventList.SelectedItem as SevereStorm;

            tileSearchInput.SetCoordinates(storm.worstLat, -storm.worstLon);

            int beforeYear = storm.month >= 8 ? storm.year : storm.year - 1;

            beforeDate.SetDate(8, beforeYear);
            afterDate.SetDate(8, beforeYear + 1); 
        }

        private int[] RectsBounds()
        {
            int x1 = (int)rects.Min(r => r.X1);
            int x2 = (int)rects.Max(r => r.X2);
            int y1 = 4095 - (int)rects.Min(r => r.Y1);
            int y2 = 4095 - (int)rects.Max(r => r.Y2);
            return [x1, x2, y1, y2];
        }

        private int[] RandomRect(int imageWidth, int imageHeight, int width, int height, int[] bounds)
        {
            while (true)
            {
                int x = r.Next(0, imageWidth);
                int y = r.Next(0, imageHeight);

                if (x + width >= imageWidth || y + height >= imageHeight) continue;

                if (x + width >= bounds[0] && x <= bounds[1] && y + height >= bounds[3] && y <= bounds[2]) continue;

                return [x, y];
            }
        }

        private void Save(object sender, RoutedEventArgs e)
        {
            SevereStorm storm = eventList.SelectedItem as SevereStorm;

            string path = "C:\\Users\\danie\\Documents\\Experiments\\Satellite\\Saved\\" + "additional_other"; //storm.location;

            System.IO.Directory.CreateDirectory(path);
            //System.IO.Directory.CreateDirectory(path + "\\before");
            //System.IO.Directory.CreateDirectory(path + "\\after");
            System.IO.Directory.CreateDirectory(path + "\\before_other");
            System.IO.Directory.CreateDirectory(path + "\\after_other");

            int fCount = Directory.GetFiles(path + "\\before_other", "*", SearchOption.AllDirectories).Length;

            var rectsEnvolope = RectsBounds();

            for (int i = 0; i < rects.Count; i++)
            {
                var rect = rects[i];

                Mat before = new(beforeImg, new OpenCvSharp.Rect((int)rect.X1, 4095 - (int)rect.Y1, 32, 32));
                Mat after = new(afterImg, new OpenCvSharp.Rect((int)rect.X1, 4095 - (int)rect.Y1, 32, 32));

                Cv2.ImWrite(path + "\\before_other\\" + (i + fCount) + ".png", before);
                Cv2.ImWrite(path + "\\after_other\\" + (i + fCount) + ".png", after);
            }

            foreach(var rect in rects)
            {
                rect.LineColor = ScottPlot.Color.FromColor(System.Drawing.Color.LightSkyBlue);
            }

            //foreach (var rect in otherRects)
            //{
            //    mainPlot.Plot.Remove(rect);
            //}

            //otherRects.Clear();

            //for (int i = 0; i < rects.Count; i++)
            //{
            //    var rect = RandomRect(beforeImg.Width, beforeImg.Height, 32, 32, rectsEnvolope);

            //    var pRect = mainPlot.Plot.Add.Rectangle(rect[0], rect[0] + 31, 4095 - 31 - rect[1], 4095 - rect[1]);
            //    pRect.FillColor = ScottPlot.Color.FromARGB(0);
            //    pRect.LineColor = ScottPlot.Color.FromColor(System.Drawing.Color.LightSkyBlue);

            //    otherRects.Add(pRect);

            //    Mat before = new(beforeImg, new OpenCvSharp.Rect(rect[0], rect[1], 32, 32));
            //    Mat after = new(afterImg, new OpenCvSharp.Rect(rect[0], rect[1], 32, 32));

            //    Cv2.ImWrite(path + "\\before_other\\" + i + ".png", before);
            //    Cv2.ImWrite(path + "\\after_other\\" + i + ".png", after);
            //}

            mainPlot.Refresh();
            //Cv2.ImWrite("C:\\Users\\danie\\Documents\\Experiments\\Satellite\\Saved\\" + storm.location + "\\" + storm.location + "_before.png", beforeImg);
            //Cv2.ImWrite("C:\\Users\\danie\\Documents\\Experiments\\Satellite\\Saved\\" + storm.location + "\\" + storm.location + "_after.png", afterImg);
        }

        List<ScottPlot.Plottables.Rectangle> rects = new();
        List<ScottPlot.Plottables.Rectangle> otherRects = new();

        private void AddMarker(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            if (e.RightButton == System.Windows.Input.MouseButtonState.Pressed)
            {
                var plt = mainPlot.Plot;
                var position = e.GetPosition(mainPlot);
                Pixel mousePixel = new(position.X * mainPlot.DisplayScale, position.Y * mainPlot.DisplayScale);

                Coordinates mouseLocation = plt.GetCoordinates(mousePixel);

                var rect = plt.Add.Rectangle((int)mouseLocation.X - 15, (int)mouseLocation.X + 16, (int)mouseLocation.Y + 16, (int)mouseLocation.Y - 15);
                rect.LineColor = ScottPlot.Color.FromColor(System.Drawing.Color.Red);
                rect.FillColor = ScottPlot.Color.FromARGB(0);

                rects.Add(rect);

                //plt.Add.Marker(mouseLocation.X, mouseLocation.Y, shape: MarkerShape.OpenSquare, color: ScottPlot.Color.FromColor(System.Drawing.Color.Red), size: 32);
                mainPlot.Refresh();

                e.Handled = true;
            }
        }

        private void RemoveRect(object sender, KeyEventArgs e)
        {
            if (rects.IsNullOrEmpty()) return;

            if (e.Key == Key.Z)
            {
                mainPlot.Plot.PlottableList.Remove(rects.Last());
                rects.RemoveAt(rects.Count - 1);
                mainPlot.Refresh();
            }
        }

        private void foundList_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            SearchResult result = (SearchResult)foundList.SelectedItem;
            tileSearchInput.SetTileIndex(result.tileX, result.tileY);

            tileSearchInput.SearchType.SelectedIndex = 1;

            Search();
        }

        private async void RunSystematicSearch(object sender, RoutedEventArgs e)
        {
            var (bMonth, bYear) = beforeDate.GetDate();
            var (aMonth, aYear) = afterDate.GetDate();

            FeatureLayer polygonLayer = PolygonSelection.GetSelectedLayer();

            var polygons = await QueuedTask.Run(() => ReadShapes<ArcGIS.Core.Geometry.Polygon>(polygonLayer));

            var results = await SystematicSearch.Search(polygons[0], bMonth, bYear, aMonth, aYear);

            results = [.. results.OrderByDescending(result => result.pixelCount)];

            foundList.Items.Clear();
            foreach (var result in results)
            {
                foundList.Items.Add(result);
            }
        }
    }
}

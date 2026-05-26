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
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using System.Formats.Asn1;
using System.Windows.Controls;

namespace Satellite_Analyzer
{

    public partial class Main : ArcGIS.Desktop.Framework.Controls.ProWindow
    {
        private bool loaded = false;
        private PlanetReader planetReader = new PlanetReader();

        private List<ImageRect> imgPlottables = [];

        private Mat beforeImg = null;
        private Mat afterImg = null;
        private Mat colorChangeMask = null;
        private Mat landcoverImg = null;
        private Mat beforeCCImg = null;
        private Mat afterCCImg = null;
        private Mat beforeUDMImg = null;
        private Mat afterUDMImg = null;
        private Mat diffImg = null;
        private Mat predAImg = null;
        private Mat predBImg = null;
        private Mat predImg = null;

        private Envelope envelope = null;
        private (double, double) worstPoint;
        private Random r = new Random();

        private SevereStorm selectedEvent = null;

        private TornadoPatchPredictor tpp;
        private TornadoPatchPredictor64_256 tpp64_256;

        public Main()
        {
            InitializeComponent();

            LoadSavedEvents();
        }

        private void LoadSavedEvents()
        {
            CanadianEventList.ItemsSource = SevereStormCA.LoadSavedEvents();
            EUEventList.ItemsSource = SevereStormEU.LoadSavedEvents();
            DownburstEventList.ItemsSource = DownburstCA.LoadSavedEvents();
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

                if (tileX == 0 || tileY == 0)
                {
                    loadingLabel.Visibility = Visibility.Hidden;
                    MessageBox.Show("Please enter a valid tile index");
                    return;
                }

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

            Cv2.MedianBlur(beforeImg, beforeImg, 5);
            Cv2.MedianBlur(afterImg, afterImg, 5);

            ByteVector tornadoPrediction = tpp.analyze(beforeImg, afterImg, beforeImg.Width, beforeImg.Height);
            predAImg = ByteVector.ToMat(tornadoPrediction, beforeImg.Size());

            tornadoPrediction = tpp64_256.analyze(beforeImg, afterImg, beforeImg.Width, beforeImg.Height);
            predBImg = ByteVector.ToMat(tornadoPrediction, beforeImg.Size());

            predImg = SystematicSearch.FloatMulNormalized(predAImg, predBImg);

            Cv2.MinMaxLoc(predImg, out double _, out double maxVal);

            scoreLabel.Content = maxVal.ToString();

            landcoverImg = await SystematicSearch.LandCoverMask(envelope, beforeImg.Size());

            Mat mask = new();
            beforeUDMImg = new Mat();
            afterUDMImg = new Mat();

            colorChangeMask = SystematicSearch.ColorChangeMask(beforeImg, afterImg);

            Cv2.BitwiseAnd(colorChangeMask, landcoverImg, mask);
            Cv2.BitwiseAnd(mask, beforeCCImg, mask);
            Cv2.BitwiseAnd(mask, afterCCImg, mask);
            //Cv2.BitwiseAnd(afterCCImg, beforeCCImg, mask);

            Cv2.BitwiseAnd(beforeImg, mask, beforeUDMImg);
            Cv2.BitwiseAnd(afterImg, mask, afterUDMImg);

            diffImg = SystematicSearch.AbsDifferenceImage(beforeUDMImg, afterUDMImg);

            imgPlottables = [MatToImageRect(beforeImg), MatToImageRect(afterImg), MatToImageRect(colorChangeMask), MatToImageRect(landcoverImg),
                             MatToImageRect(beforeCCImg), MatToImageRect(afterCCImg), MatToImageRect(beforeUDMImg), 
                             MatToImageRect(afterUDMImg), MatToImageRect(diffImg), MatToImageRect(predAImg), MatToImageRect(predBImg), MatToImageRect(predImg)];

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
            plt.PlottableList.AddRange(otherRects);

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
            tpp64_256 = new(AddinAssemblyLocation() + "\\model64_256.onnx");

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

        private void UpdateSearchParams(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            selectedEvent = (sender as ListBox).SelectedItem as SevereStorm;

            var (N, W) = selectedEvent.SearchCoords();
            tileSearchInput.SetCoordinates(N, W);

            int year = selectedEvent.SearchYear();

            beforeDate.SetDate(8, year);
            afterDate.SetDate(8, year + 1); 
        }

        private int[] RectsBounds()
        {
            if (rects.IsNullOrEmpty()) return [0, 4064, 0, 4064];
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

                if (x < width || x + width > imageWidth || y < height || y + height > imageHeight) continue;

                if (x + width >= bounds[0] && x <= bounds[1] && y + height >= bounds[3] && y <= bounds[2]) continue;

                return [x, y];
            }
        }

        bool drawOther = false;

        private void Save(object sender, RoutedEventArgs e)
        {
            SevereStorm storm = selectedEvent;

            string path = "C:\\Users\\danie\\Documents\\Experiments\\Satellite\\Saved_EU\\" + storm.Name();

            System.IO.Directory.CreateDirectory(path);
            System.IO.Directory.CreateDirectory(path + "\\before");
            System.IO.Directory.CreateDirectory(path + "\\after");
            System.IO.Directory.CreateDirectory(path + "\\before_other");
            System.IO.Directory.CreateDirectory(path + "\\after_other");
            System.IO.Directory.CreateDirectory(path + "\\before_large");
            System.IO.Directory.CreateDirectory(path + "\\after_large");
            System.IO.Directory.CreateDirectory(path + "\\before_other_large");
            System.IO.Directory.CreateDirectory(path + "\\after_other_large");

            int fCount = Directory.GetFiles(path + "\\before_other", "*", SearchOption.AllDirectories).Length;

            var rectsEnvolope = RectsBounds();

            const int rectSize = 64;
            const int largeRectSize = 256;

            for (int i = 0; i < rects.Count; i++)
            {
                var rect = rects[i];

                Mat before = new(beforeImg, new OpenCvSharp.Rect((int)rect.X1, 4095 - (int)rect.Y1, rectSize, rectSize));
                Mat after = new(afterImg, new OpenCvSharp.Rect((int)rect.X1, 4095 - (int)rect.Y1, rectSize, rectSize));

                Cv2.ImWrite(path + "\\before\\" + (i + fCount) + ".png", before);
                Cv2.ImWrite(path + "\\after\\" + (i + fCount) + ".png", after);

                Mat largeBefore = new Mat(beforeImg, new OpenCvSharp.Rect((int)rect.X1 - (largeRectSize - rectSize) / 2, 4095 - (int)rect.Y1 - (largeRectSize - rectSize) / 2, largeRectSize, largeRectSize));
                Mat largeAfter = new Mat(afterImg, new OpenCvSharp.Rect((int)rect.X1 - (largeRectSize - rectSize) / 2, 4095 - (int)rect.Y1 - (largeRectSize - rectSize) / 2, largeRectSize, largeRectSize));

                Cv2.ImWrite(path + "\\before_large\\" + (i + fCount) + ".png", largeBefore);
                Cv2.ImWrite(path + "\\after_large\\" + (i + fCount) + ".png", largeAfter);
            }

            foreach (var rect in rects)
            {
                rect.LineColor = ScottPlot.Color.FromColor(System.Drawing.Color.LightSkyBlue);
            }

            //foreach (var rect in otherRects)
            //{
            //    mainPlot.Plot.Remove(rect);
            //}

            //otherRects.Clear();

            int diff = rects.Count - otherRects.Count;

            for (int i = 0; i < diff; i++)
            {
                var rect = RandomRect(beforeImg.Width, beforeImg.Height, largeRectSize, largeRectSize, rectsEnvolope);

                var pRect = mainPlot.Plot.Add.Rectangle(rect[0], rect[0] + rectSize - 1, 4095 - (rectSize - 1) - rect[1], 4095 - rect[1]);
                pRect.FillColor = ScottPlot.Color.FromARGB(0);

                otherRects.Add(pRect);
            }

            for (int i = 0; i < otherRects.Count; i++)
            {
                var rect = otherRects[i];

                Mat before = new(beforeImg, new OpenCvSharp.Rect((int)rect.X1, 4095 - (int)rect.Y1, rectSize, rectSize));
                Mat after = new(afterImg, new OpenCvSharp.Rect((int)rect.X1, 4095 - (int)rect.Y1, rectSize, rectSize));

                Cv2.ImWrite(path + "\\before_other\\" + (i + fCount) + ".png", before);
                Cv2.ImWrite(path + "\\after_other\\" + (i + fCount) + ".png", after);

                Mat largeBefore = new Mat(beforeImg, new OpenCvSharp.Rect((int)rect.X1 - (largeRectSize - rectSize) / 2, 4095 - (int)rect.Y1 - (largeRectSize - rectSize) / 2, largeRectSize, largeRectSize));
                Mat largeAfter = new Mat(afterImg, new OpenCvSharp.Rect((int)rect.X1 - (largeRectSize - rectSize) / 2, 4095 - (int)rect.Y1 - (largeRectSize - rectSize) / 2, largeRectSize, largeRectSize));

                Cv2.ImWrite(path + "\\before_other_large\\" + (i + fCount) + ".png", largeBefore);
                Cv2.ImWrite(path + "\\after_other_large\\" + (i + fCount) + ".png", largeAfter);
            }

            foreach (var rect in otherRects)
            {
                rect.LineColor = ScottPlot.Color.FromColor(System.Drawing.Color.LightSkyBlue);
            }

            mainPlot.Refresh();
            Cv2.ImWrite(path + "\\" + storm.Name() + "_before.png", beforeImg);
            Cv2.ImWrite(path + "\\" + storm.Name() + "_after.png", afterImg);
        }

        List<ScottPlot.Plottables.Rectangle> rects = new();
        List<ScottPlot.Plottables.Rectangle> otherRects = new();

        private void AddMarker(object sender, MouseButtonEventArgs e)
        {
            const int largeRectSize = 256;
            const int smallRectSize = 64;

            if (e.RightButton == MouseButtonState.Pressed)
            {
                e.Handled = true;
                var plt = mainPlot.Plot;
                var position = e.GetPosition(mainPlot);
                Pixel mousePixel = new(position.X * mainPlot.DisplayScale, position.Y * mainPlot.DisplayScale);

                Coordinates mouseLocation = plt.GetCoordinates(mousePixel);

                if (mouseLocation.X < largeRectSize / 2 || mouseLocation.X >= 4095 - largeRectSize / 2 || mouseLocation.Y < largeRectSize / 2 || mouseLocation.Y >= 4095 - largeRectSize / 2) return;

                var rect = plt.Add.Rectangle((int)mouseLocation.X - (smallRectSize / 2 - 1), (int)mouseLocation.X + (smallRectSize / 2), (int)mouseLocation.Y + (smallRectSize / 2), (int)mouseLocation.Y - (smallRectSize / 2 - 1));
                rect.LineColor = ScottPlot.Color.FromColor(drawOther ? System.Drawing.Color.DarkOrange : System.Drawing.Color.Red);
                rect.FillColor = ScottPlot.Color.FromARGB(0);

                if (drawOther)
                {
                    otherRects.Add(rect);
                }
                else
                {
                    rects.Add(rect);
                }
                    
                //plt.Add.Marker(mouseLocation.X, mouseLocation.Y, shape: MarkerShape.OpenSquare, color: ScottPlot.Color.FromColor(System.Drawing.Color.Red), size: 64);
                mainPlot.Refresh();
            }
        }

        private void HandleKeyPressed(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Down:
                    imageList.SelectedIndex = 7;
                    break;

                case Key.Up:
                    imageList.SelectedIndex = 8;
                    break;

                case Key.Left:
                    imageList.SelectedIndex = 0;
                    break;

                case Key.Right:
                    imageList.SelectedIndex = 1;
                    break;

                case Key.Z:
                    if (drawOther)
                    {
                        if (otherRects.IsNullOrEmpty()) return;
                        mainPlot.Plot.PlottableList.Remove(otherRects.Last());
                        otherRects.RemoveAt(otherRects.Count - 1);
                    }
                    else
                    {
                        if (rects.IsNullOrEmpty()) return;
                        mainPlot.Plot.PlottableList.Remove(rects.Last());
                        rects.RemoveAt(rects.Count - 1);
                    }
                    mainPlot.Refresh();
                    break;

                case Key.S:
                    drawOther = !drawOther;
                    break;
            }

            //e.Handled = true;
        }
    }
}

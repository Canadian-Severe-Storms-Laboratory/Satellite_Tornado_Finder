using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Internal.Mapping;
using ArcGIS.Desktop.Mapping;
using Microsoft.Win32;
using Newtonsoft.Json;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using ScottPlot;
using ScottPlot.Plottables;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

using static ArcGISUtils.Utils;

namespace Satellite_Analyzer
{
    public partial class SearchWindow : ArcGIS.Desktop.Framework.Controls.ProWindow
    {
        private bool loaded = false;
        private string savePath;

        private List<ImageRect> imgPlottables;
        private List<ImageRect> nextImgPlottables;
        private List<Coordinates[]> predContours;
        private List<Coordinates[]> nextPredContours;
        private int lastResultIndex = 0;
        private bool highlightPred = true;

        public SearchWindow()
        {
            InitializeComponent();
        }

        private void UpdatePlot(object sender = null, RoutedEventArgs e = null)
        {
            if (!loaded || imgPlottables.IsNullOrEmpty()) return;

            var plt = mainPlot.Plot;
            plt.Clear();

            var imRect = imgPlottables[imageList.SelectedIndex];

            plt.PlottableList.Add(imRect);

            if (highlightPred) PlotContours(plt);

            loadingLabel.Visibility = Visibility.Hidden;

            mainPlot.Refresh();
            mainPlot.Plot.Axes.AutoScale();
        }

        private void WindowLoaded(object sender, RoutedEventArgs e)
        {
            loadingLabel.Visibility = Visibility.Visible;

            PreLoadDlls();

            SetupPlot();

            loadingLabel.Visibility = Visibility.Hidden;

            TornadoPatchPredictor tpp = new(AddinAssemblyLocation() + "\\model64.onnx");

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

        private void SetupPlot()
        {
            //mainPlot.Menu.Clear();

            //mainPlot.Menu.Add("Tornado", (plot) => { plot.Plot.Add.Marker(lastMouseLocation.X, lastMouseLocation.Y, MarkerShape.Asterisk, 30, color: ScottPlot.Color.FromColor(System.Drawing.Color.Green)); });
            //mainPlot.Menu.Add("Downburst", (plot) => { plot.Plot.Add.Marker(lastMouseLocation.X, lastMouseLocation.Y, MarkerShape.Asterisk, 30, color: ScottPlot.Color.FromColor(System.Drawing.Color.Orange)); });
            //mainPlot.Menu.Add("Unclassified", (plot) => { plot.Plot.Add.Marker(lastMouseLocation.X, lastMouseLocation.Y, MarkerShape.Asterisk, 30, color: ScottPlot.Color.FromColor(System.Drawing.Color.Blue)); });
            //mainPlot.Menu.Add("Other", (plot) => { plot.Plot.Add.Marker(lastMouseLocation.X, lastMouseLocation.Y, MarkerShape.Asterisk, 30, color: ScottPlot.Color.FromColor(System.Drawing.Color.Red)); });

            var plt = mainPlot.Plot;
            plt.Axes.SquareUnits();
            plt.Layout.Frameless();
            plt.DataBackground.Color = ScottPlot.Colors.Black;
            plt.HideAxesAndGrid();

            mainPlot.Refresh();
        }

        private void HandleKeyPressed(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Down:
                    imageList.SelectedIndex = 2;
                    break;

                case Key.Up:
                    //imageList.SelectedIndex = 3;
                    highlightPred = !highlightPred;
                    UpdatePlot();
                    break;

                case Key.Left:
                    imageList.SelectedIndex = 0;
                    break;

                case Key.Right:
                    imageList.SelectedIndex = 1;
                    break;

                case Key.Enter:
                    NextTile(null, null);
                    break;

                case Key.Z:
                    break;

                default:
                    break;
            }

            e.Handled = true;
        }

        Coordinates lastMouseLocation = new(0, 0);

        private void SaveMouseCoords(object sender, MouseButtonEventArgs e)
        {
            if (e.RightButton == System.Windows.Input.MouseButtonState.Pressed)
            {
                var plt = mainPlot.Plot;
                var position = e.GetPosition(mainPlot);
                Pixel mousePixel = new(position.X * mainPlot.DisplayScale, position.Y * mainPlot.DisplayScale);

                lastMouseLocation = plt.GetCoordinates(mousePixel);
            }
        }

        private async void foundList_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (foundList == null || foundList.SelectedItem == null) return;

            foundList.IsEnabled = false;
            nextButton.IsEnabled = false;

            if (lastResultIndex + 1 == foundList.SelectedIndex)
            {
                predContours = nextPredContours;
                imgPlottables = nextImgPlottables;
            }
            else
            {
                SearchResult r = (SearchResult)foundList.SelectedItem;
                await QueuedTask.Run(() => imgPlottables = LoadTileImages(r));
                predContours = LoadPredContours(r);
            }

            lastResultIndex = foundList.SelectedIndex;

            UpdatePlot();

            SearchResult result = (SearchResult)foundList.Items[Math.Min(foundList.SelectedIndex + 1, foundList.Items.Count - 1)];

            _ = QueuedTask.Run(() =>
            {
                MapView.Active.ZoomToAsync(resultLayers[lastResultIndex]);

                nextImgPlottables = LoadTileImages(result);
                nextPredContours = LoadPredContours(result);

                foundList.Dispatcher.Invoke(() => foundList.IsEnabled = true);
                nextButton.Dispatcher.Invoke(() => nextButton.IsEnabled = true);
            });
        }

        List<RasterLayer> resultLayers;

        private async void RunSystematicSearch(object sender, RoutedEventArgs e)
        {
            var (bMonth, bYear) = beforeDate.GetDate();
            var (aMonth, aYear) = afterDate.GetDate();

            if (bYear >= aYear)
            {
                MessageBox.Show("Before year must be less than after year");
                return;
            }

            FeatureLayer polygonLayer = PolygonSelection.GetSelectedLayer();

            if (polygonLayer == null)
            {
                MessageBox.Show("Select a polygon search area");
                return;
            }

            var polygons = await QueuedTask.Run(() => ReadShapes<ArcGIS.Core.Geometry.Polygon>(polygonLayer));

            var results = await SystematicSearch.Search(polygons[0], bMonth, bYear, aMonth, aYear);

            //results = [.. results.OrderByDescending(result => result.pixelCount)];

            foundList.Items.Clear();

            foreach (var result in results) foundList.Items.Add(result);

            savePath = SystematicSearch.GetSavePath();

            resultLayers = await FindResultLayers(results);

            foundList.SelectedIndex = 0;
        }

        private async void LoadSaveFile(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new()
            {
                Filter = "json files (*.json)|*.json|All files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == false) return;
            
            string filePath = openFileDialog.FileName;

            string jsonString = File.ReadAllText(filePath);

            dynamic jsonSaveData = JsonConvert.DeserializeObject(jsonString);

            savePath = jsonSaveData.folderPath;
            List<SearchResult> results = jsonSaveData.tiles.ToObject<List<SearchResult>>();

            foundList.Items.Clear();

            foreach (var result in results) foundList.Items.Add(result);

            resultLayers = await FindResultLayers(results);

            foundList.SelectedIndex = 0;
        }

        private async Task<List<RasterLayer>> FindResultLayers(List<SearchResult> results)
        {
            List<RasterLayer> layers = [];

            var rasterLayers = GetRasterLayers();

            bool failed = false;

            foreach (var result in results)
            {
                string imgName = $"_{result.tileX}_{result.tileY}.tif";

                bool found = false;

                foreach (var layer in rasterLayers)
                {
                    if (layer.Name == "pred" + imgName)
                    {
                        layers.Add(layer);
                        found = true;
                        break;
                    }
                }

                if (found) continue;
                
                try
                {             
                    RasterLayer rl = await QueuedTask.Run(() => { return LoadRasterLayer(savePath, "pred" + imgName); }); //LoadRasterLayer(savePath, "diff" + imgName); 
                    layers.Add(rl);
                }
                catch
                {
                    layers.Add(null);
                    failed = true;
                }  
            }

            if (failed) MessageBox.Show("Some layers failed to load");
            
            return layers;
        }

        private void PlotContours(Plot plt)
        {
            foreach (var contour in predContours)
            {
                var poly = plt.Add.Polygon(contour);
                poly.FillColor = ScottPlot.Colors.Red.WithAlpha(0.2);
                poly.LineColor = ScottPlot.Colors.Red;
                poly.LineStyle.Pattern = LinePattern.Solid;
            }
        }

        private List<Coordinates[]> LoadPredContours(SearchResult result)
        {
            string imgName = $"_{result.tileX}_{result.tileY}";
            Mat pred = Cv2.ImRead(savePath + "\\pred" + imgName + ".tif", ImreadModes.Grayscale);
            Cv2.Dilate(pred, pred, Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5)));

            List<Coordinates[]> polys = [];

            var contours = Cv2.FindContoursAsArray(pred, RetrievalModes.External, ContourApproximationModes.ApproxNone);

            foreach (var contour in contours)
            {
                Coordinates[] poly = new Coordinates[contour.Length];

                for (int i = 0; i < contour.Length; i++)
                {
                    poly[i] = new Coordinates(contour[i].X, pred.Height - contour[i].Y);
                }

                polys.Add(poly);
            }

            return polys;
        }

        private List<ImageRect> LoadTileImages(SearchResult result)
        {
            string imgName = $"_{result.tileX}_{result.tileY}";

            Mat before = Cv2.ImRead(savePath + "\\before" + imgName + ".png");
            Mat after = Cv2.ImRead(savePath + "\\after" + imgName + ".png");
            Mat diff = Cv2.ImRead(savePath + "\\diff" + imgName + ".png");
            Mat pred = Cv2.ImRead(savePath + "\\pred" + imgName + ".tif");

            List <ImageRect> imageRects = [
                MatToImageRect(before),
                MatToImageRect(after),
                MatToImageRect(diff),
                MatToImageRect(pred)
            ];

            return imageRects;
        }

        private ImageRect MatToImageRect(Mat image)
        {
            Cv2.CvtColor(image, image, ColorConversionCodes.BGR2RGBA);

            SKBitmap bmp = new();
            SKImageInfo info = new(image.Width, image.Height, SKColorType.Rgba8888, SKAlphaType.Premul);
            bool succeeded = bmp.InstallPixels(info, image.Data, info.RowBytes);

            if (!succeeded) return new ImageRect();

            return new ImageRect
            {
                Image = new(bmp),
                Rect = new(0, image.Cols, 0, image.Rows)
            };
        }

        private void NextTile(object sender, RoutedEventArgs e)
        {
            if (foundList == null || foundList.SelectedItem == null || 
                foundList.SelectedIndex == foundList.Items.Count - 1) return;

            foundList.SelectedIndex += 1;
        }
    }
}

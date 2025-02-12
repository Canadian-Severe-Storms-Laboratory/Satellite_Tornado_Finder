using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Internal.Mapping;
using ArcGIS.Desktop.Mapping;
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
        private int lastResultIndex = 0;

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

            loadingLabel.Visibility = Visibility.Hidden;

            mainPlot.Refresh();
            mainPlot.Plot.Axes.AutoScale();
        }

        private async void WindowLoaded(object sender, RoutedEventArgs e)
        {
            loadingLabel.Visibility = Visibility.Visible;

            PreLoadDlls();

            SetupPlot();

            loadingLabel.Visibility = Visibility.Hidden;

            TornadoPatchPredictor tpp = new(AddinAssemblyLocation() + "\\tornado_patch_predictor_de_norm.onnx");

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
                    imageList.SelectedIndex = 3;
                    break;

                case Key.Left:
                    imageList.SelectedIndex = 0;
                    break;

                case Key.Right:
                    imageList.SelectedIndex = 1;
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
                imgPlottables = nextImgPlottables;
            }
            else
            {
                SearchResult r = (SearchResult)foundList.SelectedItem;
                await QueuedTask.Run(() => imgPlottables = LoadTileImages(r));
            }

            lastResultIndex = foundList.SelectedIndex;

            UpdatePlot();

            SearchResult result = (SearchResult)foundList.Items[Math.Min(foundList.SelectedIndex + 1, foundList.Items.Count - 1)];

            _ = QueuedTask.Run(() =>
            {
                MapView.Active.ZoomToAsync(resultLayers[lastResultIndex]);

                nextImgPlottables = LoadTileImages(result);

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

            var polygons = await QueuedTask.Run(() => ReadShapes<ArcGIS.Core.Geometry.Polygon>(polygonLayer));

            var results = await SystematicSearch.Search(polygons[0], bMonth, bYear, aMonth, aYear);

            results = [.. results.OrderByDescending(result => result.pixelCount)];

            foundList.Items.Clear();

            foreach (var result in results) foundList.Items.Add(result);

            savePath = SystematicSearch.GetSavePath();

            resultLayers = FindResultLayers(results);

            foundList.SelectedIndex = 0;
        }

        private List<RasterLayer> FindResultLayers(List<SearchResult> results)
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

                if (!found)
                {
                    try
                    {
                        LoadRasterLayer(savePath, "diff" + imgName);
                        RasterLayer rl = LoadRasterLayer(savePath, "pred" + imgName);
                        layers.Add(rl);
                    }
                    catch
                    {
                        layers.Add(null);
                        failed = true;
                    }
                }
            }

            if (failed) MessageBox.Show("Some layers failed to load");
            
            return layers;
        }

        private List<ImageRect> LoadTileImages(SearchResult result)
        {
            string imgName = $"_{result.tileX}_{result.tileY}";

            List<ImageRect> imageRects = [
                LoadImageRect(savePath + "\\before" + imgName + ".png"),
                LoadImageRect(savePath + "\\after" + imgName + ".png"),
                LoadImageRect(savePath + "\\diff" + imgName + ".tif"),
                LoadImageRect(savePath + "\\pred" + imgName + ".tif"),
            ];

            return imageRects;
        }

        private ImageRect LoadImageRect(string filePath)
        {
            Mat image = Cv2.ImRead(filePath);
            Cv2.CvtColor(image, image, ColorConversionCodes.BGR2RGBA);

            SKBitmap bmp = new();
            SKImageInfo info = new(image.Width, image.Height, SKColorType.Rgba8888, SKAlphaType.Premul);
            bool succeeded = bmp.InstallPixels(info, image.Data, info.RowBytes);

            if (!succeeded)
            {
                return new ImageRect();
            }

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

using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Internal.Mapping;
using ArcGIS.Desktop.Mapping;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using ScottPlot;
using ScottPlot.Plottables;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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

            var plt = mainPlot.Plot;
            plt.Axes.SquareUnits();
            plt.Layout.Frameless();
            plt.DataBackground.Color = ScottPlot.Colors.Black;
            plt.HideAxesAndGrid();

            mainPlot.Refresh();

            loadingLabel.Visibility = Visibility.Hidden;

            //update to relative path...
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
                nextImgPlottables = LoadTileImages(result);

                foundList.Dispatcher.Invoke(() => foundList.IsEnabled = true);
                nextButton.Dispatcher.Invoke(() => nextButton.IsEnabled = true);
            });
        }

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

            foundList.SelectedIndex = 0;
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

            return new ImageRect
            {
                Image = new(BitmapToBytes(BitmapConverter.ToBitmap(image))),
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

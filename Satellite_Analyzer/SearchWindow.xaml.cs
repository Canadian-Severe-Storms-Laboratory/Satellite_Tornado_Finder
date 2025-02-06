using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.Internal.Mapping;
using ArcGIS.Desktop.Mapping;
using ScottPlot;
using ScottPlot.Plottables;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

using static ArcGISUtils.Utils;

namespace Satellite_Analyzer
{
    public partial class SearchWindow : ArcGIS.Desktop.Framework.Controls.ProWindow
    {
        private bool loaded = false;

        private List<ImageRect> imgPlottables = [];

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
                    break;

                default:
                    break;
            }

            e.Handled = true;
        }

        private void foundList_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (foundList.SelectedItem == null) return;

            SearchResult result = (SearchResult)foundList.SelectedItem;
            //tileSearchInput.SetTileIndex(result.tileX, result.tileY);

            //tileSearchInput.SearchType.SelectedIndex = 1;

            //Search();
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
        }

        private void NextTile(object sender, RoutedEventArgs e)
        {
            foundList.SelectedIndex += 1;
        }
    }
}

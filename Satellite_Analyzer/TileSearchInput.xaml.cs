using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Satellite_Analyzer
{
    
    public partial class TileSearchInput : UserControl
    {
        public TileSearchInput()
        {
            InitializeComponent();
        }

        public void SetCoordinates(double lat, double lon)
        {
            latBox.SetNumber(lat);
            lonBox.SetNumber(lon);
        }

        public (double, double) GetCoordinates()
        {
            return (latBox.GetNumber(), lonBox.GetNumber());
        }

        public (int, int) GetTileIndex()
        {
            return ((int)xBox.GetNumber(), (int)yBox.GetNumber());
        }

        public void SetTileIndex(int x, int y)
        {
            xBox.SetNumber(x);
            yBox.SetNumber(y);
        }

        public bool IsCoordinateSearch()
        {
            return SearchType.SelectedIndex == 0;
        }

        public bool IsTileIndexSearch()
        {
            return SearchType.SelectedIndex == 1;
        }

        private void SearchType_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (SearchType == null || CoordinateSearch == null || TileIndexSearch == null) return;

            if (SearchType.SelectedIndex == 0)
            {
                CoordinateSearch.Visibility = Visibility.Visible;
                TileIndexSearch.Visibility = Visibility.Hidden;
            }
            else
            {
                CoordinateSearch.Visibility = Visibility.Hidden;
                TileIndexSearch.Visibility = Visibility.Visible;
            }
        }
    }
}

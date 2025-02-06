using ArcGIS.Core.CIM;
using ArcGIS.Core.Data;
using ArcGIS.Core.Geometry;
using ArcGIS.Desktop.Catalog;
using ArcGIS.Desktop.Core;
using ArcGIS.Desktop.Editing;
using ArcGIS.Desktop.Extensions;
using ArcGIS.Desktop.Framework;
using ArcGIS.Desktop.Framework.Contracts;
using ArcGIS.Desktop.Framework.Dialogs;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using ArcGIS.Desktop.KnowledgeGraph;
using ArcGIS.Desktop.Layouts;
using ArcGIS.Desktop.Mapping;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Satellite_Analyzer
{
    internal class ShowSearchWindow : Button
    {

        private SearchWindow _searchwindow = null;

        protected override void OnClick()
        {
            //already open?
            if (_searchwindow != null)
                return;
            _searchwindow = new SearchWindow();
            _searchwindow.Owner = FrameworkApplication.Current.MainWindow;
            _searchwindow.Closed += (o, e) => { _searchwindow = null; };
            _searchwindow.Show();
            //uncomment for modal
            //_searchwindow.ShowDialog();
        }

    }
}

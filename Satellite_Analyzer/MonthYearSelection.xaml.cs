using System;
using System.Windows.Controls;

namespace Satellite_Analyzer
{
    public partial class MonthYearSelection : UserControl
    {
        public MonthYearSelection()
        {
            InitializeComponent();

            int year = DateTime.Now.Year;

            for (int i = year; i > 2016; i--) yearBox.Items.Add(i);
        }

        public (int, int) GetDate()
        {
            int month = monthBox.SelectedIndex + 1;
            int year = (int)yearBox.SelectedItem;
            return (month, year);
        }

        public void SetDate(int month, int year) 
        { 
            monthBox.SelectedIndex = (month - 1) % 12; 
            
            yearBox.SelectedItem = Math.Clamp(year, (int)yearBox.Items[yearBox.Items.Count - 1], (int)yearBox.Items[0]);
        }
    }
}

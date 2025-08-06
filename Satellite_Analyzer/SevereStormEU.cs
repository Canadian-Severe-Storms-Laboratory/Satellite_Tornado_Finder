using CsvHelper;
using CsvHelper.Configuration.Attributes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static ArcGISUtils.Utils;


namespace Satellite_Analyzer
{
    public class SevereStormEU : SevereStorm
    {
        [Index(0)]
        public string id { get; set; }
        [Index(1)]
        public string place { get; set; }
        [Index(2)]
        public string Country { get; set; }
        [Index(3)]
        public double N { get; set; }
        [Index(4)]
        public double E { get; set; }
        [Index(5)]
        public string date { get; set; }
        [Index(6)]
        public bool forest { get; set; }

        public override string ToString()
        {
            return $"{place}, {id.Substring(0, 2)}";
        }

        public override (double, double) SearchCoords()
        {
            return (N, -E);
        }

        public override int SearchYear()
        {
            int month = Int32.Parse(date.Substring(3, 2));
            int year = Int32.Parse(date.Substring(6, 4));

            return month >= 8 ? year : year - 1;
        }

        public static List<SevereStormEU> LoadSavedEvents()
        {
            var filePath = AddinAssemblyLocation() + "\\forest_tornadoes_eu.csv";

            var culture = new System.Globalization.CultureInfo("en-US", false);
            culture.NumberFormat.NumberDecimalDigits = 4;
            culture.NumberFormat.CurrencyDecimalDigits = 4;
            culture.NumberFormat.PercentDecimalDigits = 4;

            using var reader = new StreamReader(filePath);
            using var csv = new CsvReader(reader, culture);
            List<SevereStormEU> events = csv.GetRecords<SevereStormEU>().Where(ev => ev.forest).ToList();

            return events;
        }
    }
}

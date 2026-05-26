using CsvHelper;
using CsvHelper.Configuration.Attributes;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static ArcGISUtils.Utils;

namespace Satellite_Analyzer
{
    public class DownburstCA : SevereStorm
    {
        [Index(0)]
        public string event_name { get; set; }
        [Index(1)]
        public int year { get; set; }
        [Index(2)]
        public int month { get; set; }
        [Index(3)]
        public int day { get; set; }
        [Index(4)]
        public string province { get; set; }
        [Index(5)]
        public string efRating { get; set; }
        [Index(6)]
        public int maxSpeed { get; set; }
        [Index(7)]
        public string damageIndicator { get; set; }
        [Index(8)]
        public double worstLon { get; set; }
        [Index(9)]
        public double worstLat { get; set; }

        public override string Name()
        {
            return event_name;
        }

        public override string ToString()
        {
            return $"{event_name}, {year}";
        }

        public override (double, double) SearchCoords()
        {
            return (worstLat, -worstLon);
        }

        public override int SearchYear()
        {
            return month >= 8 ? year : year - 1;
        }

        public static List<DownburstCA> LoadSavedEvents()
        {
            var filePathCA = AddinAssemblyLocation() + "\\canadian_forest_downbursts.csv";

            var culture = new System.Globalization.CultureInfo("en-US", false);
            culture.NumberFormat.NumberDecimalDigits = 4;
            culture.NumberFormat.CurrencyDecimalDigits = 4;
            culture.NumberFormat.PercentDecimalDigits = 4;

            using var reader = new StreamReader(filePathCA);
            using var csv = new CsvReader(reader, culture);
            List<DownburstCA> events = csv.GetRecords<DownburstCA>().ToList();
            events.Sort((a, b) => a.event_name.CompareTo(b.event_name));

            return events;
        }

    }
}

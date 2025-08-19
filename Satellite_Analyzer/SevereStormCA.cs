using CsvHelper;
using CsvHelper.Configuration.Attributes;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static ArcGISUtils.Utils;

namespace Satellite_Analyzer
{
    public class SevereStormCA : SevereStorm
    {
        [Index(0)]
        public int year { get; set; }
        [Index(1)]
        public int month { get; set; }
        [Index(2)]
        public int day { get; set; }
        [Index(3)]
        public string location { get; set; }
        [Index(4)]
        public string province { get; set; }
        [Index(5)]
        public double startLat { get; set; }
        [Index(6)]
        public double startLon { get; set; }
        [Index(7)]
        public double worstLat { get; set; }
        [Index(8)]
        public double worstLon { get; set; }
        [Index(9)]
        public double endLat { get; set; }
        [Index(10)]
        public double endLon { get; set; }
        [Index(11)]
        public int efRating { get; set; }
        [Index(12)]
        public int maxSpeed { get; set; }
        [Index(13)]
        public string damageIndicator { get; set; }
        [Index(14)]
        public int maxWidth { get; set; }
        [Index(15)]
        public int maxLength { get; set; }
        [Index(16)]
        public int fromDirection { get; set; }

        public override string Name() 
        {
            return $"{location}_{year}";
        }

        public override string ToString() { 
            return $"{location}, {year}";
        }

        public override (double, double) SearchCoords()
        {
            return (worstLat, -worstLon);
        }

        public override int SearchYear()
        {
            return month >= 8 ? year : year - 1;
        }

        public static List<SevereStormCA> LoadSavedEvents()
        {
            var filePathCA = AddinAssemblyLocation() + "\\forest_tornadoes_modified.csv";

            var culture = new System.Globalization.CultureInfo("en-US", false);
            culture.NumberFormat.NumberDecimalDigits = 4;
            culture.NumberFormat.CurrencyDecimalDigits = 4;
            culture.NumberFormat.PercentDecimalDigits = 4;

            using var reader = new StreamReader(filePathCA);
            using var csv = new CsvReader(reader, culture);
            List<SevereStormCA> events = csv.GetRecords<SevereStormCA>().ToList();
            events.Sort((a, b) => a.location.CompareTo(b.location));

            return events;
        }
    }
}

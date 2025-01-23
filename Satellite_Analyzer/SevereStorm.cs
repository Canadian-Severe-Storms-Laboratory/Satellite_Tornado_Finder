
using CsvHelper;
using CsvHelper.Configuration.Attributes;

namespace Satellite_Analyzer
{
    internal class SevereStorm
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

        public override string ToString() { 
            return $"{location}, {year}";
        }
    }
}

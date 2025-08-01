using CsvHelper.Configuration.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Satellite_Analyzer
{
    internal class SevereStormEU
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
            return $"{place}, {id.Substring(0, 2)} - {forest}";
        }
    }
}

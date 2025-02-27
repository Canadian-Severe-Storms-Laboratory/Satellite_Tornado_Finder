using ArcGIS.Core.Geometry;
using Newtonsoft.Json;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using System.Windows;
using BitMiracle.LibTiff.Classic;
using System.IO;
using SkiaSharp;

namespace Satellite_Analyzer
{
    public class PlanetReader
    {
        private static readonly string API_KEY = Environment.GetEnvironmentVariable("PL_API_KEY");
        private static readonly string MOSAIC_URL = "https://api.planet.com/basemaps/v1/mosaics?_page_size=250&api_key=" + API_KEY;
        private static readonly string BASEMAP_URL = "https://link.planet.com/basemaps/v1/mosaics/{0}/quads/{1}-{2}/full?api_key=" + API_KEY;

        private HttpClient client;

        private Dictionary<(int, int), string> baseMapDict;

        public PlanetReader()
        {
            client = new HttpClient();
        }

        public async Task<(Mat, Mat, Envelope, (double, double))> FindImage(double lat, double lon, int month, int year, bool mercator = false)
        {
            Mat img;
            Mat ccMask;
            var (fx, fy) = TileIndex(lat, lon, mercator);

            int x = (int)Math.Floor(fx);
            int y = (int)Math.Ceiling(fy);

            string url = String.Format(BASEMAP_URL, baseMapDict[(month, year)], x, y);

            try
            {
                var imgTask = FetchImageTile(url);
                var maskTask = FetchUDM(url);

                await Task.WhenAll(imgTask, maskTask);

                byte[] imageBytes = imgTask.Result;
                img = Cv2.ImDecode(imageBytes, ImreadModes.Color);

                var (maskBytes, maskType) = maskTask.Result;

                ccMask = DecodeUDM(maskBytes, maskType);
            }
            catch (Exception)
            {
                return (null, null, null, (0, 0));
            }

            return (img, ccMask, TileIndexToEnvelope(x, y), ((fx - x)*img.Width, img.Height - (y - fy)*img.Height));
        }

        public async Task<(byte[], byte[], Envelope, int)> FindImage(int x, int y, int month, int year, string savePath=null)
        {
            try
            {
                string url = String.Format(BASEMAP_URL, baseMapDict[(month, year)], x, y);

                var imgTask = FetchImageTile(url);
                var maskTask = FetchUDM(url);

                await Task.WhenAll(imgTask, maskTask);

                byte[] imgBytes = imgTask.Result;
                var (maskBytes, maskType) = maskTask.Result;

                return (imgBytes, maskBytes, TileIndexToEnvelope(x, y), maskType);
            }
            catch (Exception)
            {
                return (null, null, null, 0);
            }
        }

        public async Task BuildBaseMapDict() 
        {
            baseMapDict = [];

            try
            {
                HttpResponseMessage response = await Fetch(MOSAIC_URL);
                 
                response.EnsureSuccessStatusCode();

                string responseBody = await response.Content.ReadAsStringAsync();

                dynamic json = JsonConvert.DeserializeObject(responseBody);

                foreach (dynamic mosaic in json.mosaics)
                {
                    string name = (string)mosaic.name;

                    if (!name.Contains("global_monthly")) continue;

                    var split = name.Split("_");

                    int y = Int32.Parse(split[2]);
                    int m = Int32.Parse(split[3]);

                    baseMapDict[(m, y)] = (string)mosaic.id;
                }
            }
            catch (Exception e) 
            { 
                MessageBox.Show("Failed to load Basemap URLs from Planet Labs\n\n" + e.Message);
            }
        }

        public async Task<HttpResponseMessage> Fetch(string url)
        {
            HttpResponseMessage response = null;
            Random rnd = new();

            for (int i = 0; i < 5; i++)
            {
                response = await client.GetAsync(url);

                if (response.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
                {
                    if (i < 4)
                    {
                        await Task.Delay(750 + rnd.Next(500));
                        continue;
                    }
                    
                    Console.WriteLine("Planet Server failed to respond after 5 attempts");
                }

                break;
            }

            return response;
        }

        public async Task<byte[]> FetchImageTile(string url)
        {
            HttpResponseMessage response = await Fetch(url);

            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsByteArrayAsync();
        }

        public async Task<(byte[], int)> FetchUDM(string url)
        {
            bool udm1 = false;

            HttpResponseMessage response = await Fetch(url + "&asset=ortho_udm2");

            if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                udm1 = true;
                response = await Fetch(url + "&asset=ortho_udm");

                if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
                {
                    return (null, 0);
                }
            }

            response.EnsureSuccessStatusCode();

            return (await response.Content.ReadAsByteArrayAsync(), udm1 ? 1 : 2);

            //return udm1 ? DecodeUDM1(imageBytes) : DecodeUDM2(imageBytes);
        }

        public static Mat DecodeUDM(byte[] imageBytes, int type)
        {
            if (type == 1) return DecodeUDM1(imageBytes);
            else if (type == 2) return DecodeUDM2(imageBytes);
            else return new Mat(new OpenCvSharp.Size(4096, 4096), MatType.CV_8UC3, Scalar.All(255));
        }

        public static Mat DecodeUDM1(byte[] imageBytes)
        {
            Mat img = Cv2.ImDecode(imageBytes, ImreadModes.Grayscale);

            Cv2.BitwiseAnd(img, new Scalar(2), img);
            Cv2.Threshold(img, img, 0, 255, ThresholdTypes.BinaryInv);
            Cv2.Dilate(img, img, Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5)));
            Cv2.CvtColor(img, img, ColorConversionCodes.GRAY2BGR);

            return img;
        }

        public static Mat DecodeUDM2(byte[] imageBytes)
        {
            using MemoryStream ms = new(imageBytes);
            TiffStream tiffStream = new();
            using Tiff image = Tiff.ClientOpen("in-memory", "r", ms, tiffStream);

            int samplesPerPixel = image.GetField(TiffTag.SAMPLESPERPIXEL)[0].ToInt();

            if (samplesPerPixel != 8) { 
                return new Mat(new OpenCvSharp.Size(4096, 4096), MatType.CV_8UC3, Scalar.All(255));
            }

            int width = image.GetField(TiffTag.IMAGEWIDTH)[0].ToInt();
            int height = image.GetField(TiffTag.IMAGELENGTH)[0].ToInt();

            int tileWidth = image.GetField(TiffTag.TILEWIDTH)[0].ToInt();
            int tileHeight = image.GetField(TiffTag.TILELENGTH)[0].ToInt();

            int tilesX = width / tileWidth;
            int tilesY = height / tileHeight;

            int bytesPerTile = tileWidth * tileHeight;

            byte[] imageBuffer = new byte[width * height];

            Mat img = new(new OpenCvSharp.Size(4096, 4096), MatType.CV_8UC1, Scalar.All(0));

            for (int i = 0; i < tilesY; i++)
            {
                for (int j = 0; j < tilesX; j++)
                {
                    int currentTileIndex = i * tilesX + j;

                    byte[] tileBuffer = new byte[bytesPerTile];
                    int bytesRead = image.ReadEncodedTile(currentTileIndex, tileBuffer, 0, bytesPerTile);

                    OpenCvSharp.Rect roi = new(j * tileWidth, i * tileHeight, tileWidth, tileHeight);
                    Mat imgROI = new(img, roi);
                    imgROI.SetArray(tileBuffer);
                }
            }

            Cv2.Multiply(img, new Scalar(255), img);
            Cv2.Dilate(img, img, Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5)));
            Cv2.CvtColor(img, img, ColorConversionCodes.GRAY2BGR);

            return img;
        }

        //https://link.planet.com/basemaps/v1/mosaics/ebd88eca-cd50-46da-a852-48a62730535b/quads/600-1319/full?api_key=8c99558289b84767a825f93828a25c7c&asset=ortho_udm2


        public static (double, double) LatLonToMercator(double lat, double lon) 
        {

            const double sm_a = 6378137.0;

            lat = lat * Math.PI / 180.0;
            lon = lon * Math.PI / 180.0;

            double x = sm_a * lon;
            double y = sm_a * Math.Log((Math.Sin(lat) + 1) / Math.Cos(lat));

            return (x, y);
        }

        public static Envelope TileIndexToEnvelope(int x, int y)
        {
            const double sx = 20037508.33993;
            const double sy = -20017940.460337;
            const double s = 19567.879238;

            double x1 = sx - s * x;
            double x2 = x1 - s;
            double y1 = sy + s * y;
            double y2 = y1 - s;

            MapPoint minPt = MapPointBuilderEx.CreateMapPoint(-x1, y2, SpatialReferences.WebMercator);
            MapPoint maxPt = MapPointBuilderEx.CreateMapPoint(-x2, y1, SpatialReferences.WebMercator);

            return EnvelopeBuilderEx.CreateEnvelope(minPt, maxPt);
        }

        public static (double, double) TileIndexToLatLon(int x, int y)
        {
            const double sx = 20037508.33993;
            const double sy = -20017940.460337;
            const double s = 19567.879238;

            double x1 = sx - s * x;
            double x2 = x1 - s;
            double y1 = sy + s * y;
            double y2 = y1 - s;

            return (0.5 * (y1 + y2), 0.5 * (x1 + x2));
        }

        public static (double, double) TileIndex(double lat, double lon, bool mercator = false)
        {
            var (x, y) = (lon, lat);

            if (!mercator) (x, y) = LatLonToMercator(lat, lon);
            
            const double sx = 20037508.33993;
            const double sy = -20017940.460337;
            const double s = 19567.879238;

            return ((sx - x) / s, (y - sy) / s); //floor / ceil  x * s + sy = y
        }



    }

}

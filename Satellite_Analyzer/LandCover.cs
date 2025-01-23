using System;
using ArcGIS.Core.Geometry;
using static ArcGISUtils.Utils;
using OpenCvSharp;
using ArcGIS.Core.Data.Raster;
using ArcGIS.Desktop.Mapping;
using System.Collections.Generic;
using ArcGIS.Core.Data;
using ScottPlot.Colormaps;
using System.Windows;
using ArcGIS.Desktop.Framework.Threading.Tasks;
using System.Threading.Tasks;

public static class LandCover
{

    #if DEBUG
        private static readonly string path = @"C:\Users\danie\Documents\Experiments\Satellite";
    #else
        private static readonly string path = GetProjectPath();
    #endif

    private static readonly string fileName = "landcover2020_WGS_1984.tif";
    private static Raster landcoverRaster;
    private static int width;
    private static int height;

    public enum LandCoverType
    {
        ConiferousForest = 1,
        TaigaForest = 2,
        DeciduousForest = 5,
        MixedForest = 6,
        Shrubland = 8,
        Grassland = 10,
        PolarShrubland = 11,
        PolarGrassland = 12,
        PolarBarrenland = 13,
        Wetland = 14,
        Cropland = 15,
        Barrenland = 16,
        Urban = 17,
        Water = 18,
        Snow = 19,
    }

    public static List<LandCoverType> forestTypes = [LandCoverType.ConiferousForest, LandCoverType.TaigaForest, LandCoverType.DeciduousForest, LandCoverType.MixedForest];
    public static List<LandCoverType> potentialForestTypes = [LandCoverType.ConiferousForest, LandCoverType.TaigaForest, LandCoverType.DeciduousForest, LandCoverType.MixedForest, LandCoverType.Shrubland, LandCoverType.Grassland];

    public static readonly Vec3b[] colourTable = [new(0, 0, 0), new (0, 61, 0), new(112, 156, 148), new(0, 0, 0), new(0, 0, 0), new(61, 140, 20), new(43, 117, 91), new(0, 0, 0), new(51, 138, 179), new(0, 0, 0), new(138, 207, 225), new(84, 117, 156), new(143, 212, 186), new(112, 138, 64), new(138, 163, 107), new(102, 174, 230), new(174, 171, 168), new(38, 33, 220), new(163, 112, 76), new(255, 250, 255)];

    public static async Task Initalize()
    {
        try
        {
            await QueuedTask.Run(() =>
            {
                landcoverRaster = OpenRasterDataset(path, fileName).CreateFullRaster();
                (width, height) = (landcoverRaster.GetWidth(), landcoverRaster.GetHeight());
            });
        }
        catch (Exception e)
        {
            MessageBox.Show("Failed to load landcover raster: \n" + path + "\\" + fileName + "\n\n" + e.Message);
        }
    }

    public static async Task<Mat> GetSection(Envelope envelope)
    {
        return await QueuedTask.Run(() =>
        {
            var rect = EnvolopeToImageRect(envelope, landcoverRaster);
            return LoadRasterImage<byte>(landcoverRaster, rect);
        });
    }


    public static Mat GetSection(RasterLayer referenceLayer) {
        
        RasterLayer layer = LoadRasterLayer(path, fileName);

        Raster raster = layer.GetRaster();
        Raster refRaster = referenceLayer.GetRaster();

        Envelope projectedEnvelope = (Envelope)GeometryEngine.Instance.Project(referenceLayer.QueryExtent(), layer.GetSpatialReference());

        var rect = EnvolopeToImageRect(projectedEnvelope, raster);
        int x = rect.Item1;
        int y = rect.Item2;

        Mat landCover = LoadRasterImage<byte>(raster, rect);

        Mat im = new (refRaster.GetHeight(), refRaster.GetWidth(), MatType.CV_8UC1, Scalar.All(0));

        List<MapPoint> pixelPoints = new(refRaster.GetHeight() * refRaster.GetWidth());

        for (int i = 0; i < refRaster.GetHeight(); i++)
        {
            for (int j = 0; j < refRaster.GetWidth(); j++)
            {   
                var mp = refRaster.PixelToMap(j, i);
                pixelPoints.Add(MapPointBuilderEx.CreateMapPoint(mp.Item1, mp.Item2, referenceLayer.GetSpatialReference()));
            }
        }

        Polyline pixelPolyline = PolylineBuilderEx.CreatePolyline(pixelPoints);

        var pl = (Polyline)GeometryEngine.Instance.Project(pixelPolyline, layer.GetSpatialReference());

        for (int i = 0; i < refRaster.GetHeight(); i++)
        {
            for (int j = 0; j < refRaster.GetWidth(); j++)
            {
                var mp = pl.Points[i * j + j];
                var p = raster.MapToPixel(mp.X, mp.Y);
                byte value = landCover.At<byte>(p.Item2 - y, p.Item1 - x);
                im.Set(i, j, value);
            }
        }

        MapView.Active.Map.RemoveLayer(layer);

        return im;
    }

    public static Mat TypeMask(Mat im, List<LandCoverType> types) {

        Mat mask = new(im.Size(), MatType.CV_8UC1, Scalar.All(0));

        foreach (int type in types) {
            Mat tempMask = new Mat();
            Cv2.Compare(im, type, tempMask, CmpType.EQ);
            Cv2.BitwiseOr(mask, tempMask, mask);
        }

        return mask;
    }

    public static Mat ToRGB(Mat im) {
        Mat rgb = new(im.Size(), MatType.CV_8UC3); 

        ForeachPixel(im, (i, j) => {           
            rgb.Set(i, j, colourTable[im.At<byte>(i, j)]);
        });
        
        return rgb;
    }

}


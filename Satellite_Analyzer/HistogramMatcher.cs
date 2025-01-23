using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Satellite_Analyzer
{
    public static class HistogramMatcher
    {
        public static Mat HistMatch(Mat sourceImage, Mat referenceImage)
        {
            Mat[] sourceChannels = Cv2.Split(sourceImage);
            Mat[] referenceChannels = Cv2.Split(referenceImage);
            Mat[] matchedChannels = new Mat[sourceChannels.Length];

            for (int c = 0; c < sourceChannels.Length; c++)
            {
                // Number of possible intensity values
                int histSize = 256;
                Rangef histRange = new Rangef(0, 256);

                // Calculate histograms for the source and reference images
                Mat sourceHist = new Mat();
                Mat refHist = new Mat();

                Cv2.CalcHist([sourceChannels[c]], [0], null, sourceHist, 1, [histSize], [histRange]);
                Cv2.CalcHist([referenceChannels[c]], [0], null, refHist, 1, [histSize], [histRange]);

                // Compute the cumulative distribution function (CDF) for each histogram
                float[] sourceHistData = new float[histSize];
                float[] refHistData = new float[histSize];

                sourceHist.GetArray(out sourceHistData);
                refHist.GetArray(out refHistData);

                float[] sourceCdf = new float[histSize];
                float[] refCdf = new float[histSize];

                sourceCdf[0] = sourceHistData[0];
                refCdf[0] = refHistData[0];

                for (int i = 1; i < histSize; i++)
                {
                    sourceCdf[i] = sourceCdf[i - 1] + sourceHistData[i];
                    refCdf[i] = refCdf[i - 1] + refHistData[i];
                }

                // Normalize the CDFs
                for (int i = 0; i < histSize; i++)
                {
                    sourceCdf[i] /= sourceCdf[histSize - 1];
                    refCdf[i] /= refCdf[histSize - 1];
                }

                // Create a lookup table to map pixel values from the source to the reference
                byte[] lut = new byte[histSize];

                for (int i = 0; i < histSize; i++)
                {
                    int j = 0;
                    float minDiff = Math.Abs(sourceCdf[i] - refCdf[0]);

                    for (int k = 1; k < histSize; k++)
                    {
                        float diff = Math.Abs(sourceCdf[i] - refCdf[k]);
                        if (diff < minDiff)
                        {
                            minDiff = diff;
                            j = k;
                        }
                    }
                    lut[i] = (byte)j;
                }

                // Apply the mapping to the source channel
                Mat matchedChannel = new Mat();
                Cv2.LUT(sourceChannels[c], lut, matchedChannel);
                matchedChannels[c] = matchedChannel;
            }

            // Merge the channels back into a single image
            Mat matchedImage = new Mat();
            Cv2.Merge(matchedChannels, matchedImage);

            return matchedImage;
        }

    }
}

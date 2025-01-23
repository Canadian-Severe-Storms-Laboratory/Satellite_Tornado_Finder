using System;

namespace Satellite_Analyzer
{
    public class Monitor
    {
        public bool cancelled = false;
        
        public ProgressWindow progressWindow;

        public Monitor(string title, double max, double min=0, double value=0) {
            progressWindow = new()
            {
                Title = title
            };
            progressWindow.progressBar.Maximum = max;
            progressWindow.progressBar.Minimum = min;
            progressWindow.progressBar.Value = value;

            Update(0.0);
        }

        public void Start()
        {
            progressWindow.Show();
            progressWindow.Focus();

            progressWindow.Closed += Cancel;
        }

        public void Stop()
        {
            progressWindow.Closed -= Cancel;
            progressWindow.Close();
        }

        public void Update()
        {
            Update(1.0);
        }

        public void Update(double value)
        {
            progressWindow.Dispatcher.Invoke(() =>
            {
                progressWindow.progressBar.Value += value;
                progressWindow.countLabel.Content = progressWindow.progressBar.Value.ToString("N0") + " / " + progressWindow.progressBar.Maximum.ToString("N0");
            });
        }

        private void Cancel(object sender, EventArgs e)
        {
            cancelled = true;
        }
    }
}

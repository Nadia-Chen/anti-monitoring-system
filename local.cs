using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using HTTPTest;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NAudio.WaveFormRenderer;

namespace project
{
    public partial class local : UserControl
    {
        private string selectedFile;
        private string downloadFile;
        private string foldPath;
        public string algorithm = "IAP";
        public string epsilon = "0.3";
        public string alpha = "0.01";
        public string iteration = "50";

        private readonly WaveFormRenderer waveFormRenderer;
        private readonly WaveFormRendererSettings standardSettings;


        public local()
        {
            InitializeComponent();
            waveFormRenderer = new WaveFormRenderer();
            standardSettings = new StandardWaveFormRendererSettings() { Name = "Standard" };

        }

        
        private void button1_Click(object sender, EventArgs e)
        {
            
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.InitialDirectory = @"C:\desktop";
            openFileDialog.Filter = "音频文件(*.mp3,*.wav)|*.mp3;*.wav";
            openFileDialog.Multiselect = false; //是否可以多选true=ok/false=no
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                //单个文件
                string localFileName = openFileDialog.FileName;
                externalWave_lb.Text = "当前读取文件：" + localFileName;
                externalWave_lb.AutoSize = true;
                externalWave_lb.MaximumSize = new Size(12, 0);

                selectedFile = openFileDialog.FileName;
                try
                {
                    standardSettings.TopPeakPen = new Pen(SystemColors.ControlDark);
                    standardSettings.BottomPeakPen = new Pen(SystemColors.ControlDarkDark);
                }
                catch(System.Exception e1) {
                    Console.WriteLine(e1);
                }

                RenderWaveform(1);

            }
            
        }

        private void RenderWaveform(int x)
        {
            if(x == 1)
            {
                if (selectedFile == null) return;
                var settings = GetRendererSettings();
                pictureBox1.Image = null;
                Enabled = false;
                var peakProvider = GetPeakProvider();
                Task.Factory.StartNew(() => RenderThreadFunc(peakProvider, standardSettings,1));
            }
            else if (x == 2)
            {
                if (downloadFile == null) return;
                var settings = GetRendererSettings();
                pictureBox2.Image = null;
                Enabled = false;
                var peakProvider = GetPeakProvider();
                Task.Factory.StartNew(() => RenderThreadFunc(peakProvider, standardSettings,2));
            }
            
        }

        // 图像设置
        private WaveFormRendererSettings GetRendererSettings()
        {
            // WaveFormRendererSettings settings;
            WaveFormRendererSettings settings = new WaveFormRendererSettings
            {
                TopHeight = 100,
                BottomHeight = 100,
                Width = 500
            };
            return settings;
        }

        private IPeakProvider GetPeakProvider()
        {
              return new MaxPeakProvider();
        }

        private void RenderThreadFunc(IPeakProvider peakProvider, WaveFormRendererSettings settings, int x)
        {
            Image image = null;
            if (x == 1)
            {
                try
                {
                    using (var waveStream = new AudioFileReader(selectedFile))
                    {
                        image = waveFormRenderer.Render(waveStream, peakProvider, settings);
                    }
                }
                catch (Exception e)
                {
                    MessageBox.Show(e.Message);
                }
                BeginInvoke((Action)(() => FinishedRender(image,1)));
            }
            else
            {
                try
                {
                    using (var waveStream = new AudioFileReader(downloadFile))
                    {
                        image = waveFormRenderer.Render(waveStream, peakProvider, settings);
                    }
                }
                catch (Exception e)
                {
                    MessageBox.Show(e.Message);
                }
                BeginInvoke((Action)(() => FinishedRender(image,2)));
            }
            
        }
        private void FinishedRender(Image image, int x)
        {
            if(x == 1)
            {
                pictureBox1.Image = image;
                Enabled = true;
            }
            else
            {
                pictureBox2.Image = image;
                Enabled = true;
            }
           
        }

        private void button2_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog dialog = new FolderBrowserDialog();
            dialog.Description = "请选择文件路径";

            if (dialog.ShowDialog() == DialogResult.OK)
            {
                foldPath = dialog.SelectedPath;
                DirectoryInfo theFolder = new DirectoryInfo(foldPath);
                FileInfo[] dirInfo = theFolder.GetFiles();
                label2.Text = theFolder.FullName.ToString();
            }
        }

        private void button_on_Click(object sender, EventArgs e)
        {
            
            // 上传文件
            Http http = new Http();
            NameValueCollection nameValueCollection = new NameValueCollection();
            nameValueCollection.Add("user", "admin");
            nameValueCollection.Add("target", "admin");
            nameValueCollection.Add("algorithm", algorithm);
            nameValueCollection.Add("epsilon", epsilon);
            nameValueCollection.Add("alpha", alpha);
            nameValueCollection.Add("iteration", iteration);

            try
            {
                string filename = Path.GetFileName(selectedFile);
                System.Console.WriteLine(http.HttpGet("http://182.92.226.210:8080/api/hello", "test"));
                // System.Console.WriteLine(http.HttpPostData("http://182.92.226.210:8080/api/upload", 10000, selectedFile, filename, nameValueCollection));
                JavaScriptSerializer Jss = new JavaScriptSerializer();
                string jsonObj = http.HttpPostData("http://182.92.226.210:8080/api/upload", 10000, selectedFile, filename, nameValueCollection);


                Dictionary<string, object> DicText = (Dictionary<string, object>)Jss.DeserializeObject(jsonObj);
                if (!DicText.ContainsKey("save_name"))
                    Console.WriteLine("error!");
                filename = DicText["save_name"].ToString();

                // 下载文件
                string path = foldPath + "\\test.wav";
                http.HttpDownloadFile("http://182.92.226.210:8080/api/download", path, $"account=admin&filename={filename}");

                // 可视化
                downloadFile = path;
                RenderWaveform(2);
            }
            catch
            {
                downloadFile = "D://virtualaudiocable//material//ker_test_sound.wav";
                RenderWaveform(2);
            }
           
        }

        private void radioGroup1_SelectedIndexChanged(object sender, EventArgs e)
        {
            algorithm = (string)radioGroup1.Properties.Items[radioGroup1.SelectedIndex].Value;
        }

        private void trackBarControl1_EditValueChanged(object sender, EventArgs e)
        {
            epsilon = (trackBarControl1.Value/100.0).ToString();
            label7.Text = epsilon;
        }

        private void trackBarControl2_EditValueChanged(object sender, EventArgs e)
        {
            alpha = (trackBarControl2.Value / 1000.0).ToString();
            label8.Text = alpha;
        }

        private void trackBarControl3_EditValueChanged(object sender, EventArgs e)
        {
            iteration = (trackBarControl3.Value).ToString();
            label9.Text = iteration;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            downloadFile = "D://virtualaudiocable//material//speaker_test_sound_231.wav";
            RenderWaveform(2);
        }
    }
}

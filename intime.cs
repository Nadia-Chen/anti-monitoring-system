using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using SkiaSharp;
using project;
using FFT;

namespace project
{
    public partial class intime : UserControl
    {
        public intime()
        {
            InitializeComponent();
            ScanSoundCards();
        }

        double[] LastFft;
        double MaxFft = 1;
        readonly FftProcessor FftProc = new FftProcessor();

        private WaveIn recorder;
        private BufferedWaveProvider bufferedWaveProvider;
        private SavingWaveProvider savingWaveProvider;
        private WaveOut player;
        private WaveOut externalWave;

        public static int index;

        // 开始
        private void pictureBox1_Click(object sender, EventArgs e)
        {
            darw_panel.Visible = true;
          
            pictureBox1.Visible = false;
            pictureBox2.Visible = true;
            label1.Text = "关闭连接";
            StartRecording_Click(null, null);
            checkBox2.Enabled = false;
            this.darw_panel.Paint += new PaintEventHandler(this.darw_panel_Paint);
        }
        
        // 关闭
        private void pictureBox2_Click(object sender, EventArgs e)
        {
            darw_panel.BackColor = Color.Transparent;
            darw_panel.ForeColor = Color.Transparent;
            darw_panel.Visible = false;
            pictureBox2.Visible = false;
            pictureBox1.Visible = true;
            label1.Text = "开启连接";
            StopRecording_Click(null, null);
            checkBox2.Enabled = true;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            // 全局代理
            button2.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(24)))), ((int)(((byte)(144)))), ((int)(((byte)(255)))));
            button2.ForeColor = System.Drawing.Color.White;
            button1.BackColor = System.Drawing.Color.White;
            button1.ForeColor = System.Drawing.Color.Black;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            // 智能代理
            button1.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(24)))), ((int)(((byte)(144)))), ((int)(((byte)(255)))));
            button1.ForeColor = System.Drawing.Color.White;
            button2.ForeColor = System.Drawing.Color.Black;
            button2.BackColor = System.Drawing.Color.White;
        }

        // 扫描声卡
        private void ScanSoundCards()
        {
            cbDevice.Items.Clear();
            for (int i = 0; i < NAudio.Wave.WaveIn.DeviceCount; i++)
                cbDevice.Items.Add(NAudio.Wave.WaveIn.GetCapabilities(i).ProductName);
            if (cbDevice.Items.Count > 0)
                cbDevice.SelectedIndex = 0;
            else
                MessageBox.Show("ERROR: no recording devices available");
            index = cbDevice.SelectedIndex;
        }

        private void RecorderOnDataAvailable(object sender, WaveInEventArgs waveInEventArgs)
        {
            bufferedWaveProvider.AddSamples(waveInEventArgs.Buffer, 0, waveInEventArgs.BytesRecorded);

            float max = 0;
            // interpret as 16 bit audio
            for (int index = 0; index < waveInEventArgs.BytesRecorded; index += 2)
            {
                short sample = (short)((waveInEventArgs.Buffer[index + 1] << 8) |
                                        waveInEventArgs.Buffer[index + 0]);
                // to floating point
                var sample32 = sample / 32768f;
                // absolute value 
                if (sample32 < 0) sample32 = -sample32;
                // is this the max value?
                if (sample32 > max) max = sample32;
            }
            //progressBar.Value = (int)(100 * max);
        }

        private void StartRecording_Click(object sender, EventArgs e)
        {
            // set up the recorder
            recorder = new WaveIn();
            recorder.DeviceNumber = 1;
            recorder.DataAvailable += RecorderOnDataAvailable;

            // set up our signal chain
            bufferedWaveProvider = new BufferedWaveProvider(recorder.WaveFormat);
            savingWaveProvider = new SavingWaveProvider(bufferedWaveProvider, folderaddr_lb.Text + "/temp.wav", this);

            // set up playback
            player = new WaveOut();
            player.DeviceNumber = checkBox2.Checked ? 0 : 1;
            player.Init(savingWaveProvider);

            // 播放外部音频
            WaveFileReader reader = new WaveFileReader(externalWave_lb.Text.Length > 0 ? externalWave_lb.Text : "Resources/noise.wav");
            LoopStream loop = new LoopStream(reader);
            externalWave = new WaveOut();
            externalWave.DeviceNumber = checkBox2.Checked ? 0 : 1;
            externalWave.Init(loop);
            externalWave.Play();

            // begin playback & record
            player.Play();
            recorder.StartRecording();
        }

        private void StopRecording_Click(object sender, EventArgs e)
        {
            // stop recording
            recorder.StopRecording();
            // stop playback
            player.Stop();

            externalWave.Stop();
            // finalise the WAV file
            savingWaveProvider.Dispose();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.InitialDirectory = @"C:\desktop";
            openFileDialog.Filter = "音频文件(*.mp3,*.wav)|*.mp3;*.wav";
            openFileDialog.Multiselect = false; //是否可以多选true=ok/false=no
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                //单个文件
                string localFileName = openFileDialog.FileName;
                externalWave_lb.Text = localFileName;

            }
        }


        // 文件存储位置选择
        private void button4_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog dialog = new FolderBrowserDialog();
            dialog.Description = "请选择文件路径";

            if (dialog.ShowDialog() == DialogResult.OK)
            {
                string foldPath = dialog.SelectedPath;
                DirectoryInfo theFolder = new DirectoryInfo(foldPath);
                FileInfo[] dirInfo = theFolder.GetFiles();
                folderaddr_lb.Text = theFolder.FullName.ToString();
            }
        }

        // 图像
        private void darw_panel_Paint(object sender, PaintEventArgs e)
        {
            if (LastFft is null)
                return;

            Panel panel = (Panel)sender;
            float width = panel.Width;
            float height = panel.Height;

            var g = e.Graphics;
            g.FillRectangle(new SolidBrush(Color.Transparent), new RectangleF(0, 0, width, height));

            double lastFftMax = LastFft.Max();
            MaxFft = Math.Max(MaxFft, lastFftMax);
            float[] ys = LastFft.Select(x => (float)(x / MaxFft) * height).ToArray();
            float[] xs = Enumerable.Range(0, ys.Length).Select(x => (float)x / ys.Length * width).ToArray();
            var points = LastFft.Select((mag, i) => new PointF(xs[i], height - ys[i])).ToArray();

            Pen pen = new Pen(ForeColor, 1);
            for (int i = 0; i < points.Length - 1; i++)
                g.DrawLine(pen, points[i], points[i + 1]);
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            double[] fft = FftProc.GetFft();
            if (!(fft is null))
            {
                LastFft = fft;
                darw_panel.Invalidate();
            }
        }

        private void button5_Click(object sender, EventArgs e)
        {
            // 展示波形图
            Wave f = new Wave();
            f.ShowDialog();
        }
    }

}

internal class SavingWaveProvider : IWaveProvider, IDisposable
{
    private readonly IWaveProvider sourceWaveProvider;
    private readonly WaveFileWriter writer;
    private bool isWriterDisposed;
    private intime form1;

    public SavingWaveProvider(IWaveProvider sourceWaveProvider, string wavFilePath, intime form)
    {
        this.form1 = form;
        this.sourceWaveProvider = sourceWaveProvider;
        writer = new WaveFileWriter(wavFilePath, sourceWaveProvider.WaveFormat);
    }

    public int Read(byte[] buffer, int offset, int count)
    {
        var read = sourceWaveProvider.Read(buffer, offset, count);
        if (count > 0 && !isWriterDisposed)
        {
            //Console.WriteLine(buffer[0]);
            //for (int i = 0; i < count; i++)
            //{
            //    //Console.Write(buffer[i] + " ");
            //    // 调整干扰度
            //    buffer[i] *= (byte)form1.trackBar.Value;
            //}
            //Console.WriteLine(DateTime.Now +" "+ count);
            writer.Write(buffer, offset, read);
        }
        if (count == 0)
        {
            Dispose(); // auto-dispose in case users forget
        }
        return read;
    }

    public WaveFormat WaveFormat { get { return sourceWaveProvider.WaveFormat; } }

    public void Dispose()
    {
        if (!isWriterDisposed)
        {
            isWriterDisposed = true;
            writer.Dispose();
        }
    }
}


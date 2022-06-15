using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace project
{
    public partial class Form1 : Form
    {
        public intime f1; //创建用户控件一变量
        public local f2; //创建用户控件二变量

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            f1 = new intime();    //实例化f1
            f2 = new local();    //实例化f2
            f1.Show();
            panel3.Controls.Add(f1);
        }

        private void pictureBox2_Click(object sender, EventArgs e)
        {
            label1.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(24)))), ((int)(((byte)(144)))), ((int)(((byte)(255)))));
            label2.ForeColor = System.Drawing.Color.White;
            f1.Show();   //将窗体一进行显示
            panel3.Controls.Clear();    //清空原容器上的控件
            panel3.Controls.Add(f1);    //将窗体一加入容器panel2
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {
            label2.ForeColor = System.Drawing.Color.FromArgb(((int)(((byte)(24)))), ((int)(((byte)(144)))), ((int)(((byte)(255)))));
            label1.ForeColor = System.Drawing.Color.White;
            f2.Show();   //将窗体二进行显示
            panel3.Controls.Clear();    
            panel3.Controls.Add(f2);    
        }
    }
}

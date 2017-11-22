using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;

using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Shapes;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.Storage.Pickers;
using Windows.Graphics.Imaging;

using System.Threading.Tasks;
// 빈 페이지 항목 템플릿에 대한 설명은 https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x412에 나와 있습니다.

namespace LeNet5_Test
{
    /// <summary>
    /// 자체적으로 사용하거나 프레임 내에서 탐색할 수 있는 빈 페이지입니다.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private StorageFile file;
        private BitmapDecoder bitmapDecoder;
        private Point imagePoint;
        private double scale;
        private bool isImageReady = false;
        private string port;
        private string serverIP;

        private bool udpReady = false;
        Windows.Networking.Sockets.DatagramSocket socket;
        Windows.Networking.HostName serverHost;
        public MainPage()
        {

            this.InitializeComponent();
            serverIP = "192.168.0.9";
            port = "12345";
            tbIP.PlaceholderText = serverIP;
            tbPort.PlaceholderText = port;
        }

        private async void OnOpenImageButtonClicked(object sender, RoutedEventArgs e)
        {
            var picker = new FileOpenPicker();
            picker.ViewMode = Windows.Storage.Pickers.PickerViewMode.Thumbnail;
            picker.SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.PicturesLibrary;
            picker.FileTypeFilter.Add(".jpg");
            picker.FileTypeFilter.Add(".jpeg");
            picker.FileTypeFilter.Add(".png");

            file = await picker.PickSingleFileAsync();

            if (file == null)
            {
                return;
            }
            
            SoftwareBitmap softwareBitmap;
            using (IRandomAccessStream stream = await file.OpenAsync(FileAccessMode.Read))
            {
                bitmapDecoder = await BitmapDecoder.CreateAsync(stream);

                softwareBitmap = await bitmapDecoder.GetSoftwareBitmapAsync();
            }

            if(softwareBitmap.BitmapPixelFormat != BitmapPixelFormat.Bgra8 || softwareBitmap.BitmapAlphaMode == BitmapAlphaMode.Straight)
            {
                softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
            }

            var source = new SoftwareBitmapSource();
            await source.SetBitmapAsync(softwareBitmap);

            img_total.Source = source;
            var ttv = img_total.TransformToVisual(Window.Current.Content);
            imagePoint = ttv.TransformPoint(new Point(0, 0));
            
                        
            scale = Math.Min(imageView.ActualWidth / bitmapDecoder.PixelWidth,imageView.ActualHeight/bitmapDecoder.PixelHeight);
            imagePoint.X = Math.Max(imagePoint.X - (bitmapDecoder.PixelWidth * scale) / 2, 0);
            imagePoint.Y = Math.Max(imagePoint.Y - (bitmapDecoder.PixelHeight * scale) / 2, 0);

            isImageReady = true;
        }

        private void OnRectPointerMoved(object sender, PointerRoutedEventArgs e)
        {
            OnImagePointerMoved(sender, e);
        }

        private void OnImagePointerMoved(object sender, PointerRoutedEventArgs e)
        {

            if (!isImageReady)
            {
                return;
            }
            PointerPoint point = e.GetCurrentPoint(topView);
            var x = point.Position.X;
            var y = point.Position.Y;


            TranslateTransform translateTransform = new TranslateTransform();
            
            translateTransform.X = Math.Max(0, x - rect.Width / 2);
            translateTransform.Y = Math.Max(0, y - rect.Height / 2);
            if (x + rect.Width / 2 > imagePoint.X+imageView.ActualWidth)
            {
                translateTransform.X = imagePoint.X + imageView.ActualWidth - rect.Width;
            }
            if (y + rect.Height / 2 > imagePoint.Y + imageView.ActualHeight)
            {
                translateTransform.Y = imagePoint.Y + imageView.ActualHeight - rect.Height;
            }
            Point startPoint = new Point(translateTransform.X - imagePoint.X, translateTransform.Y - imagePoint.Y);


            rect.RenderTransform = translateTransform;

            cropBitmap(startPoint);
        }
        private async void cropBitmap(Point startPoint)
        {
            uint startPointX = (uint)Math.Max(0, Math.Floor(startPoint.X ));
            uint startPointY = (uint)Math.Max(0, Math.Floor(startPoint.Y ));
            uint height = (uint)Math.Floor(rect.Height );
            uint width = (uint)Math.Floor(rect.Width );

            uint scaledWidth = (uint)Math.Floor(bitmapDecoder.PixelWidth * scale);
            uint scaledHeight = (uint)Math.Floor(bitmapDecoder.PixelHeight * scale);

            if (startPointX + width > scaledWidth)
            {
                startPointX = scaledWidth - width;
            }
            if (startPointY + height > scaledHeight)
            {
                startPointY = scaledHeight - height;
            }


            byte[] pixels = await GetPixelData(bitmapDecoder, startPointX, startPointY, width, height, scaledWidth, scaledHeight);

            WriteableBitmap cropBmp = new WriteableBitmap((int)width, (int)height);
            Stream pixStream = cropBmp.PixelBuffer.AsStream();
            pixStream.Write(pixels, 0, (int)(width * height * 4));

            img_cropped.Source = cropBmp;

            byte[] grayPixels = getGrayscaleBytes(pixels, (int)width, (int)height);

            if (udpReady)
            {
                SendUdpMessage(grayPixels);
            }
        }

        /// <summary> 
        /// Use BitmapTransform to define the region to crop, and then get the pixel data in the region 
        /// </summary> 
        /// <returns></returns> 
        async static private Task<byte[]> GetPixelData(BitmapDecoder decoder, uint startPointX, uint startPointY,
            uint width, uint height)
        {
            return await GetPixelData(decoder, startPointX, startPointY, width, height,
                decoder.PixelWidth, decoder.PixelHeight);
        }

        /// <summary> 
        /// Use BitmapTransform to define the region to crop, and then get the pixel data in the region. 
        /// If you want to get the pixel data of a scaled image, set the scaledWidth and scaledHeight 
        /// of the scaled image. 
        /// </summary> 
        /// <returns></returns> 
        async static private Task<byte[]> GetPixelData(BitmapDecoder decoder, uint startPointX, uint startPointY,
            uint width, uint height, uint scaledWidth, uint scaledHeight)
        {

            BitmapTransform transform = new BitmapTransform();
            BitmapBounds bounds = new BitmapBounds();
            bounds.X = startPointX;
            bounds.Y = startPointY;
            bounds.Height = height;
            bounds.Width = width;
            transform.Bounds = bounds;

            transform.ScaledWidth = scaledWidth;
            transform.ScaledHeight = scaledHeight;

            // Get the cropped pixels within the bounds of transform. 
            PixelDataProvider pix = await decoder.GetPixelDataAsync(
                BitmapPixelFormat.Bgra8,
                BitmapAlphaMode.Straight,
                transform,
                ExifOrientationMode.IgnoreExifOrientation,
                ColorManagementMode.ColorManageToSRgb);
            byte[] pixels = pix.DetachPixelData();
            return pixels;
        }

        private byte[] getGrayscaleBytes(byte[] rgb, int width, int height)
        {
            byte[] grayscalePixels = new byte[width * height + 2];
            byte[] grayImageByte = new byte[4 * width * height];

            for(int i = 0; i < width * height; i++)
            {
                int r = (int)rgb[4 * i + 2];
                int g = (int)rgb[4 * i + 1];
                int b = (int)rgb[4 * i];

                byte average = (byte)(255 - (r + g + b) / 3);
                unchecked
                {
                    grayImageByte[4 * i + 3] = rgb[4 * i + 3];
                    grayImageByte[4 * i + 2] = average;
                    grayImageByte[4 * i + 1] = average;
                    grayImageByte[4 * i] = average;

                    grayscalePixels[i + 1] = average;
                }
            }
            grayscalePixels[0] = (byte)'s';
            grayscalePixels[grayscalePixels.Length - 1] = (byte)'e';


            // Set Image to grayscale

            WriteableBitmap grayBmp = new WriteableBitmap((int)width, (int)height);
            Stream pixStream = grayBmp.PixelBuffer.AsStream();
            pixStream.Write(grayImageByte, 0, (int)(width * height * 4));

            grayImage.Source = grayBmp;

            // return grayscale pixels

            return grayscalePixels;
            
        }

        private void OnConnectButtonClicked(object sender, RoutedEventArgs e)
        {
            serverIP = tbIP.Text.Length==0 ? tbIP.PlaceholderText : tbIP.Text;
            port = tbPort.Text.Length == 0 ? tbPort.PlaceholderText : tbPort.Text;

            UdpClient();
            
        }

        private async void UdpClient()
        {
            try
            {
                socket = new Windows.Networking.Sockets.DatagramSocket();

                socket.MessageReceived += Socket_MessageReceived;

                serverHost = new Windows.Networking.HostName(serverIP);

                await socket.ConnectAsync(serverHost, port);

                udpReady = true;

                ConnectButton.Content = "Connected!";
            }
            catch(Exception e)
            {
                
            }
            

        }

        private async void Socket_MessageReceived(Windows.Networking.Sockets.DatagramSocket sender, Windows.Networking.Sockets.DatagramSocketMessageReceivedEventArgs e)
        {
            Stream streamIn = e.GetDataStream().AsStreamForRead();
            StreamReader streamReader = new StreamReader(streamIn);
            string message = await streamReader.ReadLineAsync();
            string[] tokens = message.Split(',');
            // Handle Message
            if (message[0] == 't' && tokens.Length==12)
            {

                string percentString =
                    "0 : " + tokens[2] + "\n" +
                    "1 : " + tokens[3] + "\n" +
                    "2 : " + tokens[4] + "\n" +
                    "3 : " + tokens[5] + "\n" +
                    "4 : " + tokens[6] + "\n" +
                    "5 : " + tokens[7] + "\n" +
                    "6 : " + tokens[8] + "\n" +
                    "7 : " + tokens[9] + "\n" +
                    "8 : " + tokens[10] + "\n" +
                    "9 : " + tokens[11] + "\n";
                    


                await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.Normal, () => {
                    //UI code here
                    resultText.Text = tokens[1];
                    percentText.Text = percentString;
                });
                
            }


        }

        private async void SendUdpMessage(char[] message)
        {
            
            Stream streamOut = (await socket.GetOutputStreamAsync(serverHost, port)).AsStreamForWrite();
            StreamWriter streamWriter = new StreamWriter(streamOut);
            await streamWriter.WriteLineAsync(message,0,message.Length);

            await streamWriter.FlushAsync();
            
        }
        private async void SendUdpMessage(byte[] message)
        {

            var outputStream = await socket.GetOutputStreamAsync(serverHost, port);
            DataWriter dataWriter = new DataWriter(outputStream);
            dataWriter.WriteBytes(message);
            await dataWriter.StoreAsync();
        }
        private void OnTerminateButtonClicked(object sender, RoutedEventArgs e)
        {
            char[] byeMessage = new char[3];
            byeMessage[0]= 'b';
            byeMessage[1]= 'y';
            byeMessage[2]= 'e';
            SendUdpMessage(byeMessage);
        }
    }


}

   

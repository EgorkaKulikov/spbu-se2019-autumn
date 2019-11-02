using System;
using System.Net;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Taks04
{
    internal class Program
    {
        private static readonly string url = "https://belkasoft.com";
        static readonly WebClient client = new WebClient();
        static readonly Regex Regex = new Regex(@"<a href=.http\S*");
        public static void Main(string[] args)
        {
            Solve();
        }

        static void Solve()
        {
            var strPageCode = client.DownloadString(url);
            var links = Regex.Matches(strPageCode);
            var sizeOfRef = links.Count;
            Console.WriteLine($"Count of valid links: {sizeOfRef}");
            foreach (var link in links)
            {
                OpenRef(link.ToString());
                System.Threading.Thread.Sleep(1000);
            }
        }

        private static async void OpenRef(string url)
        {
            url = url.Substring(9, url.Length - 10);
            var newClient = new WebClient();
            try
            {
                var strPageCode = await newClient.DownloadStringTaskAsync(url);
                Console.WriteLine("WebPage: {0} - {1}", url, strPageCode.Length);
            }
            catch
            {
                Console.WriteLine("Incorrect url");
            }
        }
    }
}
using System;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Taks04
{
    internal static class RefCounter
    {
        private const string Url = "https://belkasoft.com";
        private static readonly WebClient Client = new WebClient();
        private static readonly Regex Regex = new Regex(@"<a href=.http\S*");
        public static async Task refCount()
        {
            var strPageCode = Client.DownloadString(Url);
            var links = Regex.Matches(strPageCode);
            var sizeOfRef = links.Count;
            Console.WriteLine($"Count of valid links: {sizeOfRef}");
            foreach (var link in links)
            {
                await OpenRef(link.ToString());
            }

            await Task.CompletedTask;
        }
        
        private static async Task OpenRef(string url)
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
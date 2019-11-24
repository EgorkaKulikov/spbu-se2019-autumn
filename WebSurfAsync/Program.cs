using System;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using HtmlAgilityPack;

namespace WebSurfAsync
{
    internal class Program
    {
        private static void Main()
        {
            var inputUrl = Console.ReadLine();

            GetMentionedUrls(inputUrl).Wait();

            Console.WriteLine("That's all!");

            Console.ReadKey();
        }

        private static async Task GetMentionedUrls(string url)
        {
            var web = new HtmlWeb();
            HtmlDocument doc;
            try
            {
                doc = web.Load(url);
            }
            catch
            {
                throw new Exception("Wrong url address");
            }

            await Task
                .WhenAll(doc.DocumentNode
                    .SelectNodes("//a[@href]")
                    .Select(node =>
                        Task.Run(() => { ReadUrlAsync(node.Attributes["href"].Value); })).ToList());
        }

        private static async void ReadUrlAsync(string url)
        {
            if (!url.StartsWith("http")) return;

            var web = new WebClient();
            var pageSize = 0;

            Printer.Print(url, pageSize);

            try
            {
                pageSize = (await web.DownloadStringTaskAsync(url)).Length;
            }
            catch
            {
                pageSize = -1;
            }
            finally
            {
                Printer.Print(url, pageSize);
            }
        }
    }
}
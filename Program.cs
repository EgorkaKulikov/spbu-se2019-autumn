using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    class Program
    {
        static void Main(string[] args)
        {
            var loader = new WebLoader(Console.ReadLine());
            Task.Run(async () => await loader.Start()).GetAwaiter().GetResult();
            Console.ReadLine();
        }
    }

    public class WebLoader
    {
        public readonly string url;

        public WebLoader(string url)
        {
            this.url = url;
        }

        public async Task Start()
        {
            string pattern = @"<a href=""https?:\/\/[^""\s]+"">";

            var client = new WebClient();
            string webpage;

            try
            {
                webpage = await client.DownloadStringTaskAsync(url);
            }
            catch
            {
                Console.WriteLine("Error occured while downloading the given url <" + url + ">.");
                Console.WriteLine("Check the url and try again.");
                return;
            }

            var regex = new Regex(pattern);
            var matchedLinks = regex.Matches(webpage)
                .OfType<Match>()
                .Select(m => String.Concat(m.Value.Skip(9)))
                .Select(str => str.Substring(0, str.Length - 2))
                .ToArray();

            Console.WriteLine("Found " + matchedLinks.Length + " links in the given url <" + url + ">:");

            foreach (string link in matchedLinks)
            {
                await LoadLink(link);
            }
        }

        private async Task LoadLink(string link)
        {
            try
            {
                var client = new WebClient();
                var webpage = await client.DownloadStringTaskAsync(link);
                Console.WriteLine("   <" + link + "> with the size of " + webpage.Length + " symbols;");
            }
            catch
            {
                Console.WriteLine("   <" + link + "> cannot be loaded;");
            }
        }
    }
}

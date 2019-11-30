using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    class Program
    {
        static readonly Regex regex = new Regex(@"<a href=""https?://(\S)*"">");

        private static async Task LoadURL(string url)
        {
            var httpClient = new HttpClient();
            try
            {
                var content = await httpClient.GetStringAsync(url);
                Console.WriteLine($"{url} {content.Length}");
            }
            catch
            {
                Console.WriteLine($"Cannot download {url}");
                return;
            }
        }

        public static async Task GetContentsAsync(string uri)
        {
            var httpClient = new HttpClient();
            string content;
            try
            {
                content = await httpClient.GetStringAsync(uri);
            }
            catch
            {
                Console.WriteLine($"Cannot download {uri}");
                return;
            }

            foreach (Match link in regex.Matches(content))
            {
                string url = link.Value.Split('"')[1];
                await LoadURL(url);
            }
        }
        public static void Main(string[] args)
        {
            string baseURL = "https://math.spbu.ru";
            Task.Run(async() => await GetContentsAsync(baseURL)).GetAwaiter().GetResult();
            Console.ReadKey();
        }
    }
}

using System;
using System.Collections.Generic;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    class Program
    {
        static void Main(string[] args)
        {
            string url = Console.ReadLine();
            try
            {
                Task.Run(() => LoadAsync(url)).Wait();
            }
            catch
            {
                throw new Exception("Invalid URL");
            }
            Console.ReadKey();
        }
        

        static async Task LoadAsync(string url)
        {
            var client = new WebClient();
            var regex = new Regex(@"<a href=""(http|https)://(\S*)""");
            var matches = regex.Matches(client.DownloadString(url));
            var links = new List<string>();
            var subLoads = new List<Task>();
            
            foreach (Match match in matches)
            {
                var subUrl = match.Groups[1] + "://" + match.Groups[2];
                if (links.Contains(subUrl)) continue;
                links.Add(subUrl);
                subLoads.Add(GetInfoAsync(subUrl));
            }
            await Task.WhenAll(subLoads);
        }

        static async Task GetInfoAsync(string url)
        {
            int size = 0;
            try
            {
                var client = new WebClient();
                size = (await client.DownloadStringTaskAsync(url)).Length;
            }
            catch (Exception)
            {
                //Console.WriteLine(e.Message); 
            }
            finally
            {
                if (size != 0)
                {
                    Console.WriteLine($"{url} -- {size}");
                }
            }
        }
    }
}
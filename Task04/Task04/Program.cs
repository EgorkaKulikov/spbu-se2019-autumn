using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using System.Net;

namespace ConsoleApplication2
{
    class Program
    {
        private static string MainUrl = "https://www.wolframalpha.com/";
        private static readonly WebClient Client = new WebClient();

        static void Main(string[] args)
        {
            webTask();
        }

        static void webTask()
        {
            var page = Client.DownloadString(MainUrl);
            var rgx = new Regex(@"<a href=""http(\S*)");

            foreach (Match match in rgx.Matches(page))
            {
                var stringUrl = new string(match.Value.Skip(9).ToArray()).TrimEnd('"');
                var subUrl = new Uri(stringUrl);
                Client.DownloadStringCompleted += (sender, e) =>
                {
                    Console.WriteLine($"{stringUrl} -- {e.Result.Length}");
                };
                Client.DownloadStringAsync(subUrl);
                System.Threading.Thread.Sleep(2000);
            }
        }
    }
}

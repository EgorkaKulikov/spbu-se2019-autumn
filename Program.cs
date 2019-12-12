using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    class Program
    {
        static async Task Main(string[] args)
        {
            string URL = "https://en.wikipedia.org/wiki/Procrastination";

            await LoadAndProcessAsync(URL);

            Console.ReadKey();
        }

        static async Task LoadAndProcessAsync(string URL)
        {
            try
            {
                using (var webClient = new WebClient())
                {
                    var data = webClient.DownloadString(URL);

                    var regexPattern = @"<a href=""https?://\S*"">";
                    var regex = new Regex(regexPattern, RegexOptions.Compiled);
                    var matches = regex.Matches(data).Cast<Match>();

                    var internalUrls =
                        (
                            from Match match in matches
                            select match.Value
                                   .Replace("<a href=\"", null)
                                   .Replace("\">", null)
                        ).ToList();

                    var internalUrlLoads = new List<Task>();
                    foreach (string url in internalUrls)
                        internalUrlLoads.Add(ProcessInternalUrlAsync(url));

                    await Task.WhenAll(internalUrlLoads);
                }
            }
            catch (WebException)
            {
                Console.WriteLine("Internet doesn't love u. I'm sorry...");
            }
            catch (Exception)
            {
                Console.WriteLine("Whoops! Something is wrong!");
            }
        }

        static async Task ProcessInternalUrlAsync(string url)
        {
            try
            {
                using (var webClient = new WebClient())
                {
                    var urlData = await webClient.DownloadDataTaskAsync(url);
                    Console.WriteLine($"{url} -- {urlData.Length}");
                }
            }
            catch
            {
                Console.WriteLine($"{url} download failed!");
            }

        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    class Program
    {
        static async Task Main()
        {
            string mainUrl = Console.ReadLine();
            
            var webData = await UrlUtils.DownloadDataAsync(mainUrl);
            
            MatchCollection matches = UrlUtils.GetUrls(webData);
            List<string> urls = new List<string>();
            
            if (matches.Count > 0)
            {
                foreach (var match in matches)
                {
                    var item = match.ToString();
                    if (!urls.Contains(item))
                    {
                        urls.Add(match.ToString()); 
                    }
                }     
            }
            else
            {
                Console.WriteLine("No matches found");
                Environment.Exit(1);
            }
            
            await UrlUtils.ReadUrlsAsync(urls.Select(str => new Regex(@"https?://\S*""")
                                                .Match(str)
                                                .ToString().TrimEnd('"')));
        }
    }
}
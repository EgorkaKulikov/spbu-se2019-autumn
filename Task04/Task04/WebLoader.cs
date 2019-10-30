using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    public static class WebLoader
    {
        private const string Pattern = @"<a href=""https?:\/\/[^""]+?"">";
        
        private static readonly WebClient client = new WebClient();
        
        public static async Task Load(string reference)
        {
            try
            {
                var htmlCode = client.DownloadString(reference);
                var regex = new Regex(Pattern, RegexOptions.Compiled);
                var matchedLinks = regex.Matches(htmlCode);

                var subloads = new List<Task>();
            
                foreach (Match match in matchedLinks)
                {
                    var link = new string((match.Value).Skip(9).SkipLast(2).ToArray());
                    subloads.Add(Subload(link));
                }

                await Task.WhenAll(subloads.ToArray());
            }
            catch(Exception e)
            {
                Console.WriteLine(e);
                Console.WriteLine("Error! Failed to open main link");
            }
        }

        private static Task Subload(string reference)
        {
            try
            {
                var sz = client.DownloadStringTaskAsync(reference).Result.Length;
                Console.WriteLine(reference + " has a size of " + sz + " symbols.");
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                Console.WriteLine("Failed to open link" + reference);
            }

            return Task.CompletedTask;
        }
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    /* class WebLoader is static so as to not create an object of this class to use it.
       This does not bear any restrictions, as the only field is const anyway, and allows
       the user to avoid creating instances to use a single method. All members of this 
       class therefore must also be static, as stated in Microsoft documentation "If the 
       static keyword is applied to a class, all the members of the class must be static."
    */
    public static class WebLoader
    {
        private const string Pattern = @"<a href=""https?:\/\/[^""]+?"">";

        public static async Task Load(string reference)
        {
            try
            {
                using var client = new WebClient();
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

        private static async Task Subload(string reference)
        {
            try
            {
                using var client = new WebClient();
                var sz = (await client.DownloadStringTaskAsync(reference)).Length;
                Console.WriteLine(reference + " has a size of " + sz + " symbols.");
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
                Console.WriteLine("Failed to open link" + reference);
            }

            //return Task.CompletedTask;
        }
    }
}
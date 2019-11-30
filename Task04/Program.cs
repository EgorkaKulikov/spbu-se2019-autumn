using System;
using System.IO;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    internal class Program
    {
        private static string webpage;
        private static async Task DownloadPages()
        {
            Regex rx = new Regex(@"<a href=\""http(s)://\S+\"">",
                                 RegexOptions.Singleline| RegexOptions.IgnoreCase);
            MatchCollection matches = rx.Matches(webpage);
            for (int i = 0; i < matches.Count; i++)
            {
                await Task.Run( () =>
                {
                    string link = matches[i].Value.Substring(9, matches[i].Value.Length - 11);
                    var client = new WebClient();
                    client.DownloadFile(link, "page" + i + ".html");
                    string fileStr = File.ReadAllText("page" + i + ".html");
                    Console.WriteLine(link + ": " + fileStr.Length + " symbols");
                });
            }
        }
        
        public static void Main(string[] args)
        {
            if (args.Length < 1) Console.WriteLine("You did not provide a link");
            else
            {
                var client = new WebClient();
                webpage = client.DownloadString(args[0]);
                DownloadPages().Wait();
            }
        }
    }
}
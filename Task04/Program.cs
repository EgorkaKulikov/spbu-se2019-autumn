using System;
using System.Net;
using System.Threading.Tasks;
using System.Text.RegularExpressions;

namespace Task04
{
    class Program
    {
        static async void OpenWebsite(string url)
        {
            await Task.Run(() => DownloadWebsite(url, false));
        }

        static void DownloadWebsite(string url, bool openLinks)
        {
            WebClient client = new WebClient();
            string htmlCode;
            try
            {
                htmlCode = client.DownloadString(url);
            }
            catch (WebException)
            {
                Console.WriteLine($"Fail to open \"{url}\"");
                return;
            }
            Console.WriteLine($"{url} -- {htmlCode.Length}");
            if (openLinks)
            {
                Regex refReg = new Regex(@"<a href ?= ?""https?://[^""]+""[^>]*>");
                Regex urlReg = new Regex(@"https?://[^""]+");
                foreach (Match match in refReg.Matches(htmlCode))
                {
                    OpenWebsite(urlReg.Match(match.Value).Value);
                }
            }
        }

        static void Main(string[] args)
        {
            DownloadWebsite(Console.ReadLine(), true);
            Console.ReadKey();
        }
    }
}

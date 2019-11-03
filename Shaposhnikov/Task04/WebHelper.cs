using System;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    public class WebHelper
    {
        private readonly HttpClient _client = new HttpClient();
        private string _input = "";
        
        public async Task GetUrls(string uri)
        {
            try
            {
                _input = await _client.GetStringAsync(uri);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nA problem occured at downloading {uri}, ensure correct URL");
                Console.WriteLine("--> {0}", ex.Message);
                return;
            }

            Regex linkPattern = new Regex(@"<a href=""(http(s)?://\S+\b)"">");
            MatchCollection matches = linkPattern.Matches(_input);
            
            Console.WriteLine("Here are the links, specified on {0} page:", uri);
            foreach (Match link in matches)
            {
                await GetUriSize(link.Groups[1].Value);
            }
            Console.WriteLine("Done");
        }

        private async Task GetUriSize(string uri)
        {
            try
            {
                string input = await _client.GetStringAsync(uri);
                Console.WriteLine($"->There are {input.Length} symbols in {uri}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"<!>Couldn't download {uri} due to: {ex.Message}");
            }
        }
    }
}
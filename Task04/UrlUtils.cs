using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    static class UrlUtils
    {
        public static async Task ReadUrlsAsync(IEnumerable<string> newUrls)
        {
            var tasks = new List<Task>();
            foreach (string url in newUrls)
            {
                tasks.Add(Task.Run(async () =>
                {
                    string data = await DownloadDataAsync(url);
                    long length = data.Length;
                    if (data != "") Console.WriteLine($"URL: {url}  Length: {length}");
                }));
            }

            await Task.WhenAll(tasks);
        }
        public static async Task<string> DownloadDataAsync(string url)
        {
            var webData = "";

            try
            {
                WebRequest request = WebRequest.Create(url);
                WebResponse response = await GetResponseAsync(request);
                
                using Stream stream = response.GetResponseStream();
                using StreamReader reader = new StreamReader(stream);
                
                webData = await reader.ReadToEndAsync();

                response.Close();
            }
            catch (WebException e)
            {
                Console.WriteLine($"URL: {url} Cannot response from url");
            }
            catch (System.UriFormatException e)
            {
                Console.WriteLine(e.Message);
            }

            return webData;
        }

        static async Task<WebResponse> GetResponseAsync(WebRequest request)
        {
            return await Task.Run(request.GetResponse);
        }
        
        public static MatchCollection GetUrls(string webData)
        {
            const string pattern = @"<a\s([^>]*)href=""https?://(\S)*""";
            Regex regex = new Regex(pattern);
            return regex.Matches(webData);
        }
    }
}
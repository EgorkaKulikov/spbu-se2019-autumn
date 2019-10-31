using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    public class WebAsync
    {
        private static readonly HttpClient client = new HttpClient();

        public bool PageLoaded { get; private set; } = false;
        public string PageData { get; private set; } = "";
        public List<string> PageUrls { get; private set; } = new List<string>();
        
        public async Task GetPageData(string uri, bool printStats)
        {
            try
            {
                var responseBody = await client.GetByteArrayAsync(uri);
                PageLoaded = true;
                PageData = Encoding.UTF8.GetString(responseBody, 0, responseBody.Length);
                
                if (printStats)
                {
                    Console.WriteLine("Current page: {0}; Number of symbols: {1}"
                       , uri
                       , PageData.Length);
                }
                
            }
            catch (HttpRequestException)
            {
                Console.WriteLine("Unable to load http request, uri: {0}"
                    , uri);
            }
        }

        public List<String> MatchUrls()
        {
            var pageUrls = new List<String>();
            
            var urlRegex = new System.Text.RegularExpressions.Regex(@"<a href=""(http|https)://(\S*)""");
            var urlMatches = urlRegex.Matches(PageData);

            foreach (Match urlMatch in urlMatches)
            {
                Console.WriteLine("Matching urls.. Current data: {0}", urlMatch.Groups[2]);

                //Group 1 is http or https; group 2 specifies link text
                string completeUrl = urlMatch.Groups[1].ToString() 
                    + "://"
                    + urlMatch.Groups[2].ToString();
                pageUrls.Add(completeUrl);
            }
            return pageUrls;
        }
    }
}

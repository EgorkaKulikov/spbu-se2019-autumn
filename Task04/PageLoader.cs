using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    public class PageLoader
    {
        private static readonly HttpClient client = new HttpClient();
        
        public async Task GetPageData(string uri)
        {
            try
            {
                var responseBody = await client.GetByteArrayAsync(uri);
                var mainPageData = Encoding.UTF8.GetString(responseBody, 0, responseBody.Length);
                var subPageUrls = MatchUrls(mainPageData);
                var subPageTasks = new List<Task>();

                foreach (var url in subPageUrls)
                {
                    var subPageLoader = new PageLoader();
                    subPageTasks.Add(subPageLoader.PrintSubPageData(url));
                }
                await Task.WhenAll(subPageTasks);                
            }
            catch (HttpRequestException)
            {
                Console.WriteLine("Unable to load main page, uri: {0}", uri);
            }
        }

        public async Task PrintSubPageData(string uri)
        {
            try
            {
                var responseBody = await client.GetByteArrayAsync(uri);
                var subPageData = Encoding.UTF8.GetString(responseBody, 0, responseBody.Length);

                Console.WriteLine("Current page: {0}; Number of symbols: {1}"
                      , uri
                      , subPageData.Length);
            }
            catch (HttpRequestException)
            {
                Console.WriteLine("Unable to load sub page, uri: {0}", uri);
            }
        }

       public List<String> MatchUrls(string pageData)
        {
            var pageUrls = new List<String>();
            
            var urlRegex = new Regex(Constants.UrlRegexString);
            var urlMatches = urlRegex.Matches(pageData);

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

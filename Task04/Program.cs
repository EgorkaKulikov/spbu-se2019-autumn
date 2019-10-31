using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Task04
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            WebAsync webAsyncClient = new WebAsync();
            await webAsyncClient.GetPageData(Constants.Url, false);

            if (webAsyncClient.PageLoaded)
            {
                var pageUrls = webAsyncClient.MatchUrls();
                var webAsyncTasks = new List<Task>();
                foreach (var url in pageUrls)
                {
                    //Client for sub urls
                    var subClient = new WebAsync();
                    webAsyncTasks.Add(subClient.GetPageData(url, true));
                }
                await Task.WhenAll(webAsyncTasks);
            }
            Console.WriteLine("Finished execution, press any key to exit..");
            Console.ReadKey();
        }
    }
}

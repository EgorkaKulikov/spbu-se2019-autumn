using System;
using System.Collections.Generic;
using System.Net;
using System.Threading.Tasks;

namespace Task04
{
    class Program
    {
        private static void ProcessUrlTop(string initialUrl)
        {
            List<string> urls = HTMLInteraction.getUrls(initialUrl);
            foreach (String url in urls)
            {
                ProcessUrlLeaf(url);
            }
        }
        
        private static async void ProcessUrlLeaf(string initialUrl)
        {
            await Task.Run(() =>
                {
                    try
                    {
                        List<string> urls = HTMLInteraction.getUrls(initialUrl);
                        foreach (String url in urls)
                        {
                            Console.WriteLine(
                                "Top Url: {0}, LeafUrl: {1}, symbols: {2}",
                                initialUrl,
                                url,
                                HTMLInteraction.GetHTMLbyUrl(url).Length
                            );
                        }
                    }
                    catch (WebException)
                    {
                        Console.WriteLine("Url is not valid or connection problem happened");
                        return;
                    }
                    catch (ArgumentException)
                    {
                        Console.WriteLine("Url unsupported");
                        return;
                    }
                    catch (OperationCanceledException)
                    {
                        Console.WriteLine("No response");
                    }
                }
            );
        }

        public static void Main(string[] args)
        {
            String initialUrl = args[0];
            List<String> urls = HTMLInteraction.getUrls(initialUrl);
            Console.WriteLine("Start");
            try
            {
                List<Task> tasks = new List<Task>();
                foreach (String url in urls)
                {
                    ProcessUrlTop(url);
                }
            }
            catch (WebException)
            {
                Console.WriteLine("Url is not valid or connection problem happened");
                return;
            }
            catch (ArgumentException)
            {
                Console.WriteLine("Url unsupported");
                return;
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("No response");
            }
        }
    }
}
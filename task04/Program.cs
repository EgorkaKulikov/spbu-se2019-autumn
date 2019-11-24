using System;
using System.Collections.Generic;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04
{
    class Program
    {
        static async Task Main(String[] args)
        {
            if (args.Length != 1)
            {
                Console.WriteLine("Need exactly one argument: uri");
                Environment.Exit(1);
            }

            var uri = args[0];

            var tasks = new SortedDictionary<String, Task<(Boolean, String)>>();
            var helper = new Helper(20000);

            foreach (var link in helper.AllLinksFrom(uri))
            {
                tasks[link] = helper.DownloadString(link);
            }

            foreach (var pair in tasks)
            {
                var result = await pair.Value;

                if (result.Item1)
                {
                    Console.WriteLine($"{pair.Key}: {(await pair.Value).Item2.Length}");
                }
                else
                {
                    Console.WriteLine($"{pair.Key}: {(await pair.Value).Item2}");
                }
            }
        }
    }

    class Helper
    {
        static Regex linkRegex = new Regex("<a ([^>]* )?href=\"https?://\\S*\"");
        static Regex httpRegex = new Regex("https?://\\S*");

        private Int32 timeout;

        public Helper(Int32 timeout)
        {
            this.timeout = timeout;
        }

        public async Task<(Boolean, String)> DownloadString(String uri)
        {
            var client = new WebClient();
            var task = client.DownloadStringTaskAsync(uri);

            if (await Task.WhenAny(task, Task.Delay(timeout)) == task)
            {
                try
                {
                    return (true, await task);
                }
                catch
                {
                    return (false, "address is invalid or an error occurred while downloading");
                }
            }
            else
            {
                return (false, "timeout reached");
            }
        }

        public IEnumerable<String> AllLinksFrom(String uri)
        {
            var result = DownloadString(uri).GetAwaiter().GetResult();

            if (result.Item1)
            {
                var match = linkRegex.Match(result.Item2);

                while (match.Success)
                {
                    yield return httpRegex.Match(match.ToString()).ToString().TrimEnd('"');
                    match = match.NextMatch();
                }
            }
            else
            {
                Console.WriteLine($"{uri}: {result.Item2}");
            }

            yield break;
        }
    }
}

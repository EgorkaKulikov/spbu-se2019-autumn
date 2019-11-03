using System;
using System.Collections.Generic;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Task04 {
    class Program {
        static void Main(String[] args) {
            if (args.Length < 1) {
                Console.WriteLine("Need more arguments");
            }

            var uri = args[0];

            var tasks = new Dictionary<String, Task<String>>();

            foreach (var link in AllLinksFrom(uri)) {
                tasks[link] = GetInfoAsync(link);
            }

            foreach (var pair in tasks) {
                Console.WriteLine($"{pair.Key}: {pair.Value.GetAwaiter().GetResult()}");
            }
        }

        private static async Task<String> GetInfoAsync(String uri) {
            var client = new WebClient();
            String data;

            try {
                data = await client.DownloadStringTaskAsync(uri);
            } catch {
                return "oops";
            }

            return $"{data.Length}";
        }

        private static IEnumerable<String> AllLinksFrom(String uri) {
            var client = new WebClient();

            String data;

            try {
                data = client.DownloadString(uri);
            } catch {
                Console.WriteLine($"{uri} is invalid or an error occurred while downloading the resource.");
                yield break;
            }

            var linkRegex = new Regex("<a (([^>])* )?href=\"https?://\\S*\"");
            var httpRegex = new Regex("https?://\\S*");
            
            var match = linkRegex.Match(data);

            while (match.Success) {
                yield return httpRegex.Match(match.ToString()).ToString().TrimEnd('"');
                match = match.NextMatch();
            }

            yield break;
        }
    }
}

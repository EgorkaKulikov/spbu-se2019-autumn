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
                Console.WriteLine($"{pair.Key}: {pair.Value.Result}");
            }
        }

        private static async Task<String> GetInfoAsync(String uri) {
            var client = new WebClient();

            try {
                var symbols = await client.DownloadStringTaskAsync(uri);

                return $"{symbols.Length}";
            } catch {
            }

            return "oops";
        }

        private static IEnumerable<String> AllLinksFrom(String uri) {
            var client = new WebClient();
            var data = client.DownloadString(uri);

            var regex = new Regex("<a (([^>])* )?href=\"https?://\\S*\"", RegexOptions.Compiled);
            var regex2 = new Regex("https?://\\S*");
            var matches = regex.Matches(data);

            foreach (var match in matches) {
                yield return regex2.Match(match.ToString()).ToString().TrimEnd('"');
            }

            yield break;;
        }
    }
}

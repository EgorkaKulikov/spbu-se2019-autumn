using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;

namespace Task04
{
    public class HTMLInteraction
    {
        public static String GetHTMLbyUrl(String urlAddress)
        {
            HttpWebRequest request = (HttpWebRequest) WebRequest.Create(urlAddress);
            HttpWebResponse response = (HttpWebResponse) request.GetResponse();

            if (response.StatusCode == HttpStatusCode.OK)
            {
                Stream receiveStream = response.GetResponseStream();
                StreamReader readStream = null;

                if (response.CharacterSet == null)
                {
                    readStream = new StreamReader(receiveStream);
                }
                else
                {
                    readStream = new StreamReader(receiveStream, Encoding.GetEncoding(response.CharacterSet));
                }

                string data = readStream.ReadToEnd();

                response.Close();
                readStream.Close();

                return data;
            }
            return "";
        }

        public static List<String> getUrls(String initialUrl)
        {
            String page = "";
            try
            {
                page = HTMLInteraction.GetHTMLbyUrl(initialUrl);
            }
            catch (WebException)
            {
                Console.WriteLine("Url {0} is not valid", initialUrl);
                return Enumerable.Empty<string>().ToList();
            }

            var anchorRegex = new Regex(
                @"<a ([^>])*href=""(http|https)://(\S*)""([^>])*>");
            var urlRegex = new Regex(
                @"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?");

            List<String> urls = new List<string>();
            foreach (Match match in anchorRegex.Matches(page))
            {
                String anchor = match.Groups[0].Value;
                Match linkMatch = urlRegex.Match(anchor);
                String linkedUrl = linkMatch.Groups[0].Value;
                if (linkedUrl == "")
                {
                    continue;
                }

                urls.Add(linkedUrl);
            }

            return urls;
        }
    }
} 
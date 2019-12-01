using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using System.Net;
using System.Text.RegularExpressions;

namespace Task04
{
    public class htmlCodeObserver
    {
            private string getHTMLcode(string URL)
            {
                try
                {
                    var request = (HttpWebRequest)WebRequest.Create(URL);
                    var response = (HttpWebResponse)request.GetResponse();

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
                            try
                            {
                                readStream = new StreamReader(receiveStream, Encoding.GetEncoding(response.CharacterSet));
                            }
                            catch (System.ArgumentException)
                            {
                                Console.Write("This charset is not provided");
                                return null;
                            }

                        }

                        return readStream.ReadToEnd();
                    }
                }
                catch (System.Net.WebException)
                {
                    Console.Write("Server may be out of order");
                    return null;
                }
                throw new Exception("HttpStatusCode isn't OK");
            }

            private async Task writeSymbolsAsync(List<string> URLs)
            {
                foreach (string url in URLs)
                {
                    await Task.Run(() => Console.Write(url + ' ' + getHTMLcode(url).Length + "\n"));
                }
            }

            public async Task symbolsAmount(string url)
            {
                string HTMLcode = getHTMLcode(url);
                var matches = Regex.Matches(HTMLcode, "<a href=\"https://.*\">");
                List<string> urls = new List<string>();

                foreach (Match match in matches)
                {
                    urls.Add(Regex.Match(match.Value, "https://[^\"]*").Value);
                }
                await writeSymbolsAsync(urls);
            }
    }
}
using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using System.Net;
using System.Text.RegularExpressions;

namespace Task04
{
    class Program
    {
        static string getHTMLcode(string URL)
        {

            try
            {
                HttpWebRequest request = (HttpWebRequest)WebRequest.Create(URL);
                HttpWebResponse response = (HttpWebResponse)request.GetResponse();

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
                            return "This charset is not provided";
                        }

                    }

                    return readStream.ReadToEnd();
                }
            }
            catch (System.Net.WebException)
            {
                Console.Write("Server may be out of order");
                return "Server may be out of order";
            }
            throw new Exception("HttpStatusCode isn't OK");
        }

        static async void writeSymbolsAsync(List<string> URLs)
        {
            foreach (string ur in URLs)
            {
                await Task.Run(() => Console.Write(ur + ' ' + getHTMLcode(ur).Length + "\n"));
            }
        }

        static void symbols_amount(string URL)
        {
            string HTMLcode = getHTMLcode(URL);
            var matches = Regex.Matches(HTMLcode, "<a href=\"https://.*\">");
            List<string> URLs = new List<string>();

            foreach (Match match in matches)
            {
                URLs.Add(Regex.Match(match.Value, "https://[^\"]*").Value);
            }
            writeSymbolsAsync(URLs);
        }

        static void Main(string[] args)
        {
            string srcURL;
            srcURL = "https://spbu.ru/";
            symbols_amount(srcURL);
            //end prgramm when get symbol
            Console.ReadKey();
        }
    }
}

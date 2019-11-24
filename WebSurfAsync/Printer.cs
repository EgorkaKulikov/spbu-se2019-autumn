using System;
using System.Collections.Generic;
using System.Threading;

namespace WebSurfAsync
{
    public class Printer
    {
        private static readonly SortedDictionary<string, int> UrlInfoPositions = new SortedDictionary<string, int>();
        private static int _lastCursorPosition;
        private static readonly Mutex WriteUrlInfo = new Mutex();

        public static void Print(string url, int pageSize)
        {
            WriteUrlInfo.WaitOne();

            if (!UrlInfoPositions.ContainsKey(url))
            {
                UrlInfoPositions.Add(url, _lastCursorPosition + 1);
                _lastCursorPosition += 1;
            }

            Console.SetCursorPosition(0, UrlInfoPositions[url]);

            var truncatedUrlMessage = url.Substring(0, Math.Min(url.Length, 100));
            string pageSizeMessage;

            switch (pageSize)
            {
                case 0:
                    pageSizeMessage = "awaiting..";
                    break;
                case -1:
                    pageSizeMessage = "download failed";
                    break;
                default:
                    pageSizeMessage = pageSize.ToString();
                    break;
            }

            Console.WriteLine($"{truncatedUrlMessage,100} - {pageSizeMessage,15}");

            WriteUrlInfo.ReleaseMutex();
        }
    }
}
using System;
using System.Net;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Taks04
{
    internal static class Program
    {
        public static async Task Main(string[] args)
        {
            await RefCounter.refCount();
        }
    }
}
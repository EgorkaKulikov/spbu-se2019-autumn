using System;

namespace Task04
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            var helper = new htmlCodeObserver();
            string srcURL = "https://spbu.ru/";
            helper.symbolsAmount(srcURL);
            //end prgramm when get symbol
            Console.ReadKey();
        }
    }
}
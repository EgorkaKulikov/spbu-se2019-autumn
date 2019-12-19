using System;
using System.Threading.Tasks;

namespace Task03
{
    public class Misc
    {
        public static Task[] ConcatTasks(Task[] x, Task[] y)
        {
            if (x == null) throw new ArgumentNullException(nameof(x));
            if (y == null) throw new ArgumentNullException(nameof(y));
            int oldLen = x.Length;
            Array.Resize<Task>(ref x, x.Length + y.Length);
            Array.Copy(y, 0, x, oldLen, y.Length);
            return x;
        }
    }
}
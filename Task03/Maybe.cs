namespace Task03
{
    class Maybe<T>
    {
        T value;

        Maybe(T val)
        {
            value = val;
        }

        public static Maybe<T> Nothing = null;

        public static Maybe<T> Just(T val) => new Maybe<T>(val);

        public T getValue() => value;
    }
}
